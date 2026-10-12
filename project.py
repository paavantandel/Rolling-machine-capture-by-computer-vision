import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt
import json
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import sys
import yaml
import time
import os
from collections import deque
import threading

# Matplotlib for embedded chart
import matplotlib
matplotlib.use("Agg")  # safe backend for embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

VIDEO_PATH = r"rolling_test.mp4"   # path to your 25-second clip



# =========================
# Tk root and base window
# =========================
root = tk.Tk()
root.title("Smart Rolling Station Monitor (Async Popups)")
root.geometry("1400x800")
root.minsize(1100, 700)

# =========================
# Non-blocking Popup Manager
# =========================
class AsyncPopupManager:
    """
    Non-blocking popups:
    - No wait_window, no blocking loops.
    - Topmost Toplevel, closes on button click.
    - Can pass callbacks for Retry/Cancel.
    """
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.active = {}  # key -> Toplevel

    def _show(self, key, title, body, buttons):
        # If already showing this popup, do nothing
        if key in self.active and self.active[key] and tk.Toplevel.winfo_exists(self.active[key]):
            return

        top = tk.Toplevel(self.parent)
        top.title(title)
        top.transient(self.parent)
        top.attributes("-topmost", True)
        top.resizable(False, False)

        frm = ttk.Frame(top, padding=12)
        frm.pack(fill="both", expand=True)

        lbl = ttk.Label(frm, text=body, justify="left", wraplength=520)
        lbl.pack(pady=(0, 12))

        btns_frame = ttk.Frame(frm)
        btns_frame.pack(fill="x")

        # Center relative to parent
        def center():
            top.update_idletasks()
            px = self.parent.winfo_rootx()
            py = self.parent.winfo_rooty()
            pw = self.parent.winfo_width()
            ph = self.parent.winfo_height()
            tw = top.winfo_width()
            th = top.winfo_height()
            x = px + (pw - tw) // 2
            y = py + (ph - th) // 2
            top.geometry(f"+{x}+{y}")

        # Register as active
        self.active[key] = top

        # Button factory
        for text, callback in buttons:
            ttk.Button(btns_frame, text=text, command=lambda cb=callback, k=key: self._on_click(k, cb)).pack(side="right", padx=6)

        center()

        # When user closes window via [X]
        def on_close():
            self._dismiss(key)
        top.protocol("WM_DELETE_WINDOW", on_close)

    def _on_click(self, key, cb):
        # Dismiss then run callback (if any)
        self._dismiss(key)
        if callable(cb):
            try:
                cb()
            except Exception as e:
                print(f"[Popup callback error] {e}")

    def _dismiss(self, key):
        top = self.active.get(key)
        if top and tk.Toplevel.winfo_exists(top):
            try:
                top.destroy()
            except Exception:
                pass
        self.active.pop(key, None)

    def error(self, key, title, body, ok_cb=None):
        self._show(key, title, body, buttons=[("OK", ok_cb)])

    def warning(self, key, title, body, ok_cb=None):
        self._show(key, title, body, buttons=[("OK", ok_cb)])

    def retry_cancel(self, key, title, body, retry_cb=None, cancel_cb=None):
        self._show(key, title, body, buttons=[("Cancel", cancel_cb), ("Retry", retry_cb)])

    def dismiss(self, key):
        self._dismiss(key)

POPUPS = AsyncPopupManager(root)

# =========================
# Layout: 50% video | 50% info+graph
# =========================
main_split = ttk.Panedwindow(root, orient="horizontal")
main_split.pack(fill="both", expand=True)

left_video_frame = ttk.LabelFrame(main_split, text="Live Video")
video_label = tk.Label(left_video_frame, bg="#111111")
video_label.pack(fill="both", expand=True, padx=6, pady=6)

right_split = ttk.Panedwindow(main_split, orient="vertical")

info_frame = ttk.LabelFrame(right_split, text="Detection Info")
condition_label = tk.Label(info_frame, text="Condition: -", font=("Segoe UI", 22, "bold"))
condition_label.pack(anchor="w", padx=10, pady=(10, 4))
top_class_label = tk.Label(info_frame, text="Top Class: - (0.00)", font=("Segoe UI", 14))
top_class_label.pack(anchor="w", padx=10, pady=(0, 6))

ttk.Label(info_frame, text="Detections (current frame):", font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
columns = ("Class", "Confidence")
det_table = ttk.Treeview(info_frame, columns=columns, show="headings", height=10)
det_table.heading("Class", text="Class")
det_table.heading("Confidence", text="Confidence")
det_table.column("Class", width=180, anchor="w")
det_table.column("Confidence", width=120, anchor="center")
det_table.pack(fill="both", expand=True, padx=10, pady=(0, 10))

chart_frame = ttk.LabelFrame(right_split, text="Activity (Last 1 Hour)")
fig = Figure(figsize=(8, 5), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Activity (Counts per Minute)")
ax.set_xlabel("Time (last 60 min)")
ax.set_ylabel("Frame count")
ax.grid(True, alpha=0.25)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
canvas = FigureCanvasTkAgg(fig, master=chart_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill="both", expand=True, padx=6, pady=6)

main_split.add(left_video_frame, weight=1)
main_split.add(right_split, weight=1)
right_split.add(info_frame, weight=1)
right_split.add(chart_frame, weight=2)

# =========================
# Globals + State
# =========================
CONFIG_PATH = "config_MQTT_v1.yaml"
model = None
client = None
cap = None

last_condition = None
first_no_detection_time = None
first_no_load_time = None
last_detection_time = None
last_send_time = None
last_capture_time_nd = None
last_capture_time_nl = None
camera_error_popup_shown = False
retry_thread_running = False  # avoid spawning multiple retry threads

events = deque(maxlen=3600)  # last ~1 hour
CONDITIONS = ["Rolling", "Edge Breaking", "No Load", "No detection"]
COLORS = {
    "Rolling": (44, 160, 44),
    "Edge Breaking": (31, 119, 180),
    "No Load": (255, 127, 14),
    "No detection": (214, 39, 40),
    "Unknown": (127, 127, 127),
    "Model not loaded": (148, 103, 189),
    "Camera Problem": (140, 86, 75),
}

# Config placeholders
mqtt_broker = mqtt_port = mqtt_username = mqtt_password = mqtt_topic = None
yolo_model_path = None
detection_interval = 1.0
data_sending_interval = 2.0
conf_threshold = 0.35
camera_url = None
assetCode = assetName = assetType = None

# =========================
# Helpers
# =========================
def cond_color(cond):
    b, g, r = COLORS.get(cond, (0, 0, 0))
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

def render_frame_to_label(img_bgr, label_widget: tk.Label):
    """Resize frame to fit widget while preserving aspect ratio (non-blocking)."""
    try:
        h, w = img_bgr.shape[:2]
        lw = label_widget.winfo_width()
        lh = label_widget.winfo_height()
        if lw <= 1 or lh <= 1:
            target_w, target_h = w, h
        else:
            aspect = w / h
            if lw / lh > aspect:
                target_h = lh
                target_w = int(lh * aspect)
            else:
                target_w = lw
                target_h = int(lw / aspect)
        img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))
        label_widget.img_tk = img_tk  # prevent GC
        label_widget.config(image=img_tk)
    except Exception as e:
        print(f"Render error: {e}")

def update_activity_graph():
    now = datetime.now()
    start = now - timedelta(hours=1)
    minute_marks = [start.replace(second=0, microsecond=0) + timedelta(minutes=i) for i in range(61)]
    counts_by_condition = {cond: [0] * len(minute_marks) for cond in CONDITIONS}
    for ts, cond in list(events):
        if ts < start:
            continue
        minute_ts = ts.replace(second=0, microsecond=0)
        try:
            idx = minute_marks.index(minute_ts)
        except ValueError:
            continue
        if cond in counts_by_condition:
            counts_by_condition[cond][idx] += 1

    ax.clear()
    ax.set_title("Activity (Counts per Minute)")
    ax.set_xlabel("Time (last 60 min)")
    ax.set_ylabel("Frame count")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))

    def bgr_to_hex(bgr): b, g, r = bgr; return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    series = [counts_by_condition[cond] for cond in CONDITIONS]
    colors = [bgr_to_hex(COLORS[cond]) for cond in CONDITIONS]
    times_num = mdates.date2num(minute_marks)

    try:
        ax.stackplot(times_num, series, labels=CONDITIONS, colors=colors, alpha=0.85)
        ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    except Exception:
        for cond, data, color in zip(CONDITIONS, series, colors):
            ax.plot(minute_marks, data, label=cond, color=color)
        ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    canvas.draw_idle()
    root.after(5000, update_activity_graph)

# =========================
# MQTT helpers
# =========================
def send_mqtt_signal(status: str):
    if client is not None:
        try:
            message_data = {
                "dataType": "system",
                "timeStamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "assetCode": assetCode,
                "assetName": assetName,
                "assetType": assetType,
                "systemStatus": status
            }
            client.publish(mqtt_topic, json.dumps(message_data))
            print(f"MQTT System Signal: {json.dumps(message_data)}")
        except Exception as e:
            print(f"Error publishing MQTT message: {str(e)}")
    else:
        print("MQTT client is not connected; skipping send.")

# >>> CHANGE START: disable system messages completely
def start_end_signal(status: str):
    # System messages disabled: do nothing
    return
# >>> CHANGE END

# =========================
# Camera helpers (threaded retry)
# =========================
RETRY_INTERVAL_MS = 2000
FRAME_INTERVAL_MS = 20


def open_camera_quick():
    """Open video file instead of camera."""
    try:
        _cap = cv2.VideoCapture(VIDEO_PATH)

        if not _cap.isOpened():
            return None

        return _cap
    except Exception:
        return None
    
def schedule_camera_retry():
    """Spawn a background thread to try reopening the camera (non-blocking)."""
    global retry_thread_running
    if retry_thread_running:
        return
    retry_thread_running = True

    def worker():
        cap_try = open_camera_quick()
        # Post result back to UI thread
        def on_result():
            global cap, camera_error_popup_shown, retry_thread_running
            retry_thread_running = False
            if cap_try is not None:
                cap = cap_try
                camera_error_popup_shown = False  # re-arm future popups
                # Dismiss any existing camera popup
                POPUPS.dismiss("camera_error")
                # Show a small connected notification (non-blocking)
                POPUPS.warning("camera_reconnected", "Camera Reconnected", "Camera connection restored.")
                # Auto-dismiss reconnection toast after 2s
                root.after(2000, lambda: POPUPS.dismiss("camera_reconnected"))
            else:
                # Schedule another attempt
                root.after(RETRY_INTERVAL_MS, schedule_camera_retry)
        root.after(0, on_result)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

# =========================
# Main update loop
# =========================
def update_frame():
    global cap, camera_error_popup_shown
    global last_condition, last_detection_time, first_no_detection_time, first_no_load_time
    global last_send_time, last_capture_time_nd, last_capture_time_nl

    current_time = datetime.now()

    # Camera Problem → show label and ensure retry is ongoing
    if cap is None:
        condition_label.config(text="Condition: Camera Problem", fg=cond_color("Camera Problem"))
        top_class_label.config(text="Top Class: - (0.00)")
        # Show non-blocking popup once
        if not camera_error_popup_shown:
            camera_error_popup_shown = True
            POPUPS.retry_cancel(
                key="camera_error",
                title="Camera Error",
                body=("Failed to open/read the camera.\n\n"
                      "• Check camera power, cable/IP, and OS permissions.\n"
                      "• Click 'Retry' to try opening again, or 'Cancel' to continue.\n"
                      "The app will auto-retry in the background."),
                retry_cb=schedule_camera_retry,
                cancel_cb=lambda: None
            )
        # Keep retrying in the background
        schedule_camera_retry()
        # Send telemetric with flag 0
        if last_send_time is None or (current_time - last_send_time) >= timedelta(seconds=data_sending_interval):
            if client is not None:
                message_data = {
                    "dataType": "telemetric",
                    "timeStamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "assetCode": assetCode,
                    "assetName": assetName,
                    "assetType": assetType,
                    "flag": 0,
                    "currentStatus": "Camera Problem"
                }
                client.publish(mqtt_topic, json.dumps(message_data))
                print(f"MQTT Update: {json.dumps(message_data)}")
            last_send_time = current_time
        events.append((current_time, "Camera Problem"))
        root.after(RETRY_INTERVAL_MS, update_frame)
        return

    # Read frame

# Restart video when finished
# Read frame

# Read frame from video
    ret, img = cap.read()

    # Loop video when it finishes
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, img = cap.read()

        if not ret:
            root.after(FRAME_INTERVAL_MS, update_frame)
            return

    rendered = img
    # Detection interval
    if last_detection_time is None or (current_time - last_detection_time) >= timedelta(seconds=detection_interval):
        if model is not None:
            try:
                results = model(img, conf=conf_threshold)
                result = results[0]
                labels = result.names
                boxes = getattr(result, "boxes", None)

                if boxes is not None and boxes.cls is not None and boxes.conf is not None:
                    pred_classes = boxes.cls.cpu().numpy()
                    conf_scores = boxes.conf.cpu().numpy()
                else:
                    pred_classes = np.array([])
                    conf_scores = np.array([])

                present_classes = [labels[int(cls)] for cls in pred_classes] if pred_classes.size > 0 else []

                # Sort by confidence desc
                if conf_scores.size > 0:
                    idx_sorted = np.argsort(-conf_scores)
                    top_pairs = [(labels[int(pred_classes[i])], float(conf_scores[i])) for i in idx_sorted[:12]]
                else:
                    top_pairs = []

                # Update table
                for item in det_table.get_children():
                    det_table.delete(item)
                if top_pairs:
                    for cls_name, score in top_pairs:
                        det_table.insert("", "end", values=(cls_name, f"{score:.2f}"))
                    top_class_label.config(text=f"Top Class: {top_pairs[0][0]} ({top_pairs[0][1]:.2f})")
                else:
                    top_class_label.config(text="Top Class: - (0.00)")

                # Condition classification
                if 'Roller' in present_classes and 'EB' in present_classes and 'Rolling' not in present_classes:
                    condition_text = "Edge Breaking"
                elif 'Roller' in present_classes and 'EB' in present_classes and 'Rolling' in present_classes:
                    condition_text = "Rolling"
                elif 'Roller' in present_classes and 'Rolling' in present_classes:
                    condition_text = "Rolling"
                elif 'Roller' in present_classes and 'EB' not in present_classes and 'Rolling' not in present_classes:
                    condition_text = "No Load"
                else:
                    condition_text = "No detection"

                condition_label.config(text=f"Condition: {condition_text}", fg=cond_color(condition_text))
                events.append((current_time, condition_text))

                # Image capture windows
                if condition_text == "No detection":
                    if first_no_detection_time is None:
                        first_no_detection_time = current_time
                    if current_time - first_no_detection_time <= timedelta(minutes=2):
                        if last_capture_time_nd is None or (current_time - last_capture_time_nd) >= timedelta(seconds=30):
                            for _ in range(5):
                                rc, frame = cap.read()
                                if rc and frame is not None:
                                    cv2.imwrite(os.path.join(save_folder, f"no_detection_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"), frame)
                            last_capture_time_nd = current_time
                    else:
                        first_no_detection_time = None
                        last_capture_time_nd = None

                elif condition_text == "No Load":
                    if first_no_load_time is None:
                        first_no_load_time = current_time
                    if current_time - first_no_load_time <= timedelta(minutes=2):
                        if last_capture_time_nl is None or (current_time - last_capture_time_nl) >= timedelta(seconds=30):
                            for _ in range(5):
                                rc, frame = cap.read()
                                if rc and frame is not None:
                                    cv2.imwrite(os.path.join(save_folder, f"no_load_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"), frame)
                            last_capture_time_nl = current_time
                    else:
                        first_no_load_time = None
                        last_capture_time_nl = None
                else:
                    first_no_detection_time = None
                    last_capture_time_nd = None
                    first_no_load_time = None
                    last_capture_time_nl = None

                last_condition = condition_text

                # Telemetry with flag=1 since camera is connected and condition detected
                if last_send_time is None or (current_time - last_send_time) >= timedelta(seconds=data_sending_interval):
                    if client is not None:
                        message_data = {
                            "dataType": "telemetric",
                            "timeStamp": current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            "assetCode": assetCode,
                            "assetName": assetName,
                            "assetType": assetType,
                            "flag": 1,
                            "currentStatus": condition_text
                        }
                        client.publish(mqtt_topic, json.dumps(message_data))
                        print(f"MQTT Update: {json.dumps(message_data)}")
                    last_send_time = current_time

                last_detection_time = current_time

            except Exception as e:
                print(f"Error during detection: {str(e)}")
                condition_label.config(text="Condition: Unknown", fg=cond_color("Unknown"))
                events.append((current_time, "Unknown"))
                rendered = img
        else:
            condition_label.config(text="Condition: Model not loaded", fg=cond_color("Model not loaded"))
            events.append((current_time, "Model not loaded"))
            rendered = img
    else:
        rendered = img

    # Show frame
    render_frame_to_label(rendered, video_label)
    root.after(FRAME_INTERVAL_MS, update_frame)

# =========================
# Clean shutdown
# =========================
def on_closing():
    start_end_signal("Application Closed")
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    try:
        if client is not None:
            client.loop_stop()
            client.disconnect()
    except Exception:
        pass
    # Dismiss popups
    POPUPS.dismiss("camera_error")
    POPUPS.dismiss("camera_reconnected")
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# =========================
# Initialization inside UI thread (non-blocking)
# =========================
save_folder = datetime.now().strftime("captured_images_%d-%m-%Y")
os.makedirs(save_folder, exist_ok=True)

def init_app():
    global mqtt_broker, mqtt_port, mqtt_username, mqtt_password, mqtt_topic
    global yolo_model_path, detection_interval, data_sending_interval, conf_threshold, camera_url
    global assetCode, assetName, assetType
    global model, client, cap

    # Load config
    try:
        with open("config_MQTT_v1.yaml", "r") as file:
            config = yaml.safe_load(file)
        mqtt_broker = config["mqtt_broker"]
        mqtt_port = config["mqtt_port"]
        mqtt_username = config["mqtt_username"]
        mqtt_password = config["mqtt_password"]
        mqtt_topic = config["mqtt_topic"]
        yolo_model_path = config["yolo_model"]
        detection_interval = config["detection_interval"]
        data_sending_interval = config["data_sending_interval"]
        conf_threshold = config["conf"]
        camera_url = config.get("camera_url", None)
        assetCode = config["assetCode"]; assetName = config["assetName"]; assetType = config["assetType"]
    except Exception as e:
        POPUPS.error("config_error", "Config Error", f"Failed to load config:\n{str(e)}\n\nThe application cannot start.")
        # Let UI remain responsive; user can close the app
        return

    # Load YOLO (continue on fail)
    try:
        model = YOLO(yolo_model_path)
    except Exception as e:
        POPUPS.warning("model_warning", "Model Error", f"Failed to load YOLO model:\n{str(e)}\nDetection will be disabled.")
        model = None

    # MQTT (continue on fail; use API v2 to avoid deprecation warnings)
    try:
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    except Exception:
        client = mqtt.Client()
    try:
        client.username_pw_set(mqtt_username, mqtt_password)
        client.connect(mqtt_broker, mqtt_port, 5)
        client.loop_start()
    except Exception as e:
        POPUPS.warning("mqtt_warning", "MQTT Error", f"Failed to connect to MQTT broker:\n{str(e)}\nContinuing without MQTT.")
        client = None

    start_end_signal("Application Started")

    # Camera: start a background open attempt, show popup (non-blocking)
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        POPUPS.error("video_error", "Video Error", "Failed to open video file.")
        return

# Start loops
update_frame()
update_activity_graph()

# Kick off init AFTER UI is visible, so popups run inside Tk loop
root.after(100, init_app)
root.mainloop()
