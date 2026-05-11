# VQA-Cam

A visual question answering surveillance system powered by [ViLT](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa). Ask questions about camera images and trigger actions when answers match – running locally on Android (Termux), Raspberry Pi, or any Linux system.

---

## Features

- **Loop mode** – continuously captures images and runs VQA at a configurable interval
- **Match & trigger** – define a question, an expected answer, and a shell command to run on match
- **Parallel capture** – next photo is taken while the model is still processing the current one
- **Alarm gallery** – matched images are saved with metadata (question, answer, timestamp, follow-up Q&A)
- **Follow-up questions** – ask additional questions about saved alarm images directly in the UI
- **Web UI** – mobile-friendly interface with live monitor, gallery grouped by day, and captures view
- **CLI tool** – single-shot VQA for use in shell scripts, motionEye, cron jobs, etc.

---

## Components

| File | Description |
|------|-------------|
| `server.py` | Flask + Socket.IO backend |
| `vqa_cam.html` | Web UI (serve via nginx or Flask) |
| `vqa_cli.py` | Single-shot CLI tool |

---

## Requirements

**Linux / Raspberry Pi:**
```
flask
flask-cors
flask-socketio
python-socketio
python-engineio
eventlet
transformers
torch
Pillow
```
```bash
pip install flask flask-cors flask-socketio python-socketio python-engineio eventlet transformers torch Pillow
```

**Termux (Android):**
```bash
pkg install "python-torch*"
pip install flask flask-cors flask-socketio python-socketio python-engineio eventlet transformers Pillow --break-system-packages
```

---

## Server

```bash
python3 server.py
```

The server runs on `http://127.0.0.1:5666` by default. The ViLT model is downloaded automatically on first run and cached locally in `~/vilt-vqa`.

### Configuration (top of `server.py`)

```python
IMAGE_PATH   = "~/vqa-scripts/images/foto.jpg"
ALARM_DIR    = "~/vqa-scripts/alarms"
CAPTURE_DIR  = "~/vqa-scripts/captures"
SOUND_DIR    = "~/vqa-scripts/sounds"
HTML_PATH    = "~/vqa-scripts/vqa_cam.html"
PORT         = 5666
CAPTURE_LIMIT = 100   # max number of captures kept on disk
```

---

## Web UI

### nginx setup (Termux)

Install nginx:
```bash
pkg install nginx
```

Create the vqa-cam directory and copy the client:
```bash
mkdir -p /data/data/com.termux/files/usr/share/nginx/html/vqa-cam
cp vqa_cam.html /data/data/com.termux/files/usr/share/nginx/html/vqa-cam/vqa_cam.html
```

Start nginx:
```bash
nginx
```

The UI is then available at:
```
http://127.0.0.1:8080/vqa-cam/vqa_cam.html
```

### Alternative – Flask serves the UI directly

Set `HTML_PATH` in `server.py` to the full path of `vqa_cam.html` and access it at `http://127.0.0.1:5666/`.

---

### Monitor tab
- Take a single photo and ask your configured questions
- Start/stop the loop
- Live answers with match highlighting
- Follow-up questions on the current image

### Gallery tab
- Alarm images grouped by day
- Original question + answer shown below each image
- Follow-up Q&A stored and displayed per image
- All captures in a separate scrollable section

### Settings tab
- Loop interval
- Sound on alarm
- Save alarm images
- Save all captures (with automatic limit)
- Server URL

---

## Loop with vqa_cli.py

Use `vqa_cli.py` in a shell loop for continuous monitoring without the server:

```bash
# Termux
while true; do
  termux-camera-photo ~/tmp/foto.jpg
  python3 vqa_cli.py "Do you see a human?" "yes" \
    --image ~/tmp/foto.jpg \
    --cmd "termux-notification -t Alarm"
  sleep 10
done

# Raspberry Pi
while true; do
  libcamera-still -o /tmp/foto.jpg --nopreview -t 1
  python3 vqa_cli.py "Is the door open?" "yes" \
    --image /tmp/foto.jpg \
    --cmd "curl http://homeassistant.local/webhook/door"
  sleep 5
done
```

---

## CLI Tool

Single-shot VQA for use in scripts, motionEye, cron, etc.

```bash
python3 vqa_cli.py "Do you see a human?" "yes" --image foto.jpg
```

```bash
# With command on match
python3 vqa_cli.py "Do you see a human?" "yes" \
  --image foto.jpg \
  --cmd "termux-notification -t Alarm"

# Exit code only (for shell scripts)
if python3 vqa_cli.py "Do you see a human?" "yes" --image foto.jpg --quiet; then
    echo "Match!"
fi
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `question` | ✓ | Question to ask about the image |
| `match` | ✓ | Word that must appear in the answer |
| `--image` | ✓ | Path to image file |
| `--cmd` | | Shell command to run on match |
| `--model-dir` | | Local model directory (default: `~/vilt-vqa`) |
| `--model-id` | | HuggingFace model ID |
| `--quiet` | | Suppress output, use exit codes only |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Match |
| `1` | No match |
| `2` | Error |

---

## Termux / Android

```bash
# Take photo and run VQA
termux-camera-photo ~/tmp/foto.jpg && \
python3 vqa_cli.py "Do you see a human?" "yes" \
  --image ~/tmp/foto.jpg \
  --cmd "termux-notification -t Alarm"
```

## Raspberry Pi

```bash
libcamera-still -o /tmp/foto.jpg && \
python3 vqa_cli.py "Do you see a human?" "yes" \
  --image /tmp/foto.jpg \
  --cmd "echo alarm triggered"
```

## motionEye integration

In motionEye, set the **Motion Triggered Command** to:

```bash
python3 /path/to/vqa_cli.py "Do you see a human?" "yes" --image %f --cmd "curl http://yourserver/alert"
```

---

VQA-Cam uses [dandelin/vilt-b32-finetuned-vqa](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa), a Vision-and-Language Transformer fine-tuned on VQA v2. The model takes an image and a natural language question and returns the most likely answer from its label set.

Since the model has a fixed answer vocabulary, questions should be phrased to elicit answers the model knows – typically simple words like `yes`, `no`, color names, object names, numbers.

**Tip:** Test your questions first:
```bash
termux-camera-photo ~/tmp/test.jpg
python3 vqa_cli.py "what do you see?" "anything" --image ~/tmp/test.jpg
```

---

## License

MIT
