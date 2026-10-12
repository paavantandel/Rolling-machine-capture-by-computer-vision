# Smart Rolling Station Monitor

Desktop computer-vision app that watches a rolling station (live camera or video file), classifies operating conditions with YOLO, and publishes telemetry over MQTT. The Tkinter UI shows live video, current detections, and a one-hour activity chart.

## Features

- YOLO object detection on incoming frames (`Roller`, `EB`, `Rolling`)
- Condition classification: **Rolling**, **Edge Breaking**, **No Load**, **No detection**, plus **Camera Problem** / **Model not loaded**
- Split UI: live video (left), detection table (right), stacked activity chart (last 60 minutes)
- Non-blocking popups for config, model, MQTT, and camera/video errors
- MQTT telemetric messages with asset metadata and `currentStatus`
- Snapshot capture during **No detection** and **No Load** windows
- Background retry when the video/camera source fails

## Project layout

```
CV_Project/
├── project.py              # Main application
├── config_MQTT_v1.yaml     # MQTT, model, intervals, asset fields
├── best.pt                 # YOLO weights (place this next to project.py)
└── rolling_test.mp4        # Demo video (optional; path is set in project.py)
```

On startup the app also creates a dated folder such as `captured_images_12-10-2026/` for snapshots.

## Requirements

- Python 3.8+
- A YOLO weights file (`best.pt` by default)
- Video file or camera/RTSP source
- MQTT broker (optional; the UI still runs if the broker is unreachable)

Install dependencies:

```bash
pip install opencv-python numpy ultralytics paho-mqtt pillow pyyaml matplotlib
```

## Configuration

Copy the example config, then fill in your broker, camera, and asset values:

```bash
# Windows
copy config_MQTT_v1.example.yaml config_MQTT_v1.yaml

# macOS / Linux
cp config_MQTT_v1.example.yaml config_MQTT_v1.yaml
```

Edit `config_MQTT_v1.yaml`:

| Key | Purpose |
|-----|---------|
| `mqtt_broker`, `mqtt_port`, `mqtt_username`, `mqtt_password`, `mqtt_topic` | MQTT connection and publish topic |
| `yolo_model` | Path to YOLO weights (default `best.pt`) |
| `conf` | Detection confidence threshold |
| `camera_url` | RTSP or camera URL (loaded in config; current code plays `VIDEO_PATH` in `project.py`) |
| `detection_interval` | Seconds between YOLO inference runs |
| `data_sending_interval` | Seconds between MQTT telemetric publishes |
| `assetCode`, `assetName`, `assetType` | Fields included in every MQTT payload |

The video source is currently hardcoded in `project.py`:

```python
VIDEO_PATH = r"rolling_test.mp4"
```

Point that to another file, or switch `open_camera_quick()` / `init_app()` to `cv2.VideoCapture(camera_url)` for a live RTSP camera.

## Run

```bash
python project.py
```

The window opens immediately. Config, model, MQTT, and video are initialized shortly after so error popups stay non-blocking.

## Condition logic

Detections on the current frame are mapped as follows:

| Detected classes | Condition |
|------------------|-----------|
| `Roller` + `EB`, no `Rolling` | Edge Breaking |
| `Roller` + `EB` + `Rolling` | Rolling |
| `Roller` + `Rolling` | Rolling |
| `Roller` only | No Load |
| Anything else | No detection |

## MQTT payloads

Telemetric messages (when the client is connected):

```json
{
  "dataType": "telemetric",
  "timeStamp": "2026-10-12 06:30:00.000",
  "assetCode": "HXX-XXX",
  "assetName": "Faccin",
  "assetType": "Smart Rolling Station",
  "flag": 1,
  "currentStatus": "Rolling"
}
```

- `flag` is `1` when a frame was read successfully, `0` for **Camera Problem**.
- `currentStatus` is the classified condition string.

Application start/stop system messages are currently disabled in code.

## Snapshots

If **No detection** or **No Load** persists, the app captures bursts of 5 frames every 30 seconds for up to 2 minutes and writes JPEGs into the dated `captured_images_*` folder.

## Notes

- Keep MQTT credentials in `config_MQTT_v1.yaml` out of version control if you share this repo.
- YOLO inference runs on the UI thread at `detection_interval`; lower the interval or use a smaller model if the UI feels sluggish.
- The activity chart refreshes every 5 seconds and stacks counts per minute for the four main conditions.
