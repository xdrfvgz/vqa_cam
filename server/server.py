# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# vqa_server.py

import os
import sys
import contextlib
import io
import base64
import subprocess
import shutil
import threading
import time
import json
from datetime import datetime

import eventlet
import eventlet.tpool

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import ViltProcessor, ViltForQuestionAnswering, logging
logging.set_verbosity_error()

from PIL import Image
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit

DEFAULT_MODEL    = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_MODELDIR = os.path.expanduser("~/vilt-vqa")
IMAGE_PATH       = os.path.expanduser("~/vqa-scripts/images/foto.jpg")
IMAGE_PATH_NEXT  = os.path.expanduser("~/vqa-scripts/images/foto_next.jpg")
ALARM_DIR        = os.path.expanduser("~/vqa-scripts/alarms")
CAPTURE_DIR      = os.path.expanduser("~/vqa-scripts/captures")
SOUND_DIR        = os.path.expanduser("~/vqa-scripts/sounds")
HTML_PATH        = os.path.expanduser("~/vqa-scripts/vqa_surveillance.html")
HOST             = "127.0.0.1"
PORT             = 5666
CAPTURE_LIMIT    = 100

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

processor = None
model = None

# Loop-State
loop_thread = None
loop_running = False
loop_lock = threading.Lock()
loop_config = {
    "interval": 5,
    "questions": [],
    "save": True,
    "save_all": False,
    "sound": True,
    "soundfile": "default_alarm_sound.wav",
}
loop_stats = {
    "total": 0,
    "alarms": 0,
    "saved": 0,
    "last_capture": None,
    "started_at": None,
}

# Capture-Zähler
capture_index_lock = threading.Lock()

def get_next_capture_index():
    if not os.path.exists(CAPTURE_DIR):
        return 1
    files = [f for f in os.listdir(CAPTURE_DIR) if f.lower().endswith(".jpg")]
    if not files:
        return 1
    nums = []
    for f in files:
        try:
            nums.append(int(f.split("_")[0]))
        except:
            pass
    return max(nums) + 1 if nums else 1

capture_index = get_next_capture_index()


def load_model(model_id, model_dir):
    global processor, model
    if not os.path.exists(model_dir):
        print("Modell wird heruntergeladen: " + model_id)
        processor = ViltProcessor.from_pretrained(model_id)
        model = ViltForQuestionAnswering.from_pretrained(model_id)
        processor.save_pretrained(model_dir)
        model.save_pretrained(model_dir, safe_serialization=True)
        print("Gespeichert in: " + model_dir)
    else:
        print("Lade Modell aus: " + model_dir)
        with contextlib.redirect_stderr(io.StringIO()):
            processor = ViltProcessor.from_pretrained(model_dir, local_files_only=True)
            model = ViltForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    print("Modell bereit.")


def take_photo_to(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    before = os.path.getmtime(path) if os.path.exists(path) else 0
    #result = subprocess.run(["termux-camera-photo -c 2", path])
    result = subprocess.run(f"termux-camera-photo -c 2 {path}", shell=True)
    if result.returncode != 0:
        return False
    after = os.path.getmtime(path) if os.path.exists(path) else 0
    return after > before


def take_photo():
    return take_photo_to(IMAGE_PATH)


def run_vqa(question):
    if not os.path.exists(IMAGE_PATH):
        return None
    img = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(img, question, return_tensors="pt")
    def _infer():
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_id = outputs.logits.argmax(-1).item()
        return model.config.id2label[predicted_id]
    return eventlet.tpool.execute(_infer)


def save_alarm_file(question="", answer=""):
    if not os.path.exists(IMAGE_PATH):
        return None
    os.makedirs(ALARM_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = ts + ".jpg"
    dest = os.path.join(ALARM_DIR, filename)
    shutil.copy2(IMAGE_PATH, dest)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
    }
    with open(os.path.join(ALARM_DIR, ts + ".json"), "w") as f:
        json.dump(meta, f)
    return filename


def save_capture_file():
    global capture_index
    if not os.path.exists(IMAGE_PATH):
        return None
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    files = sorted([
        f for f in os.listdir(CAPTURE_DIR) if f.lower().endswith(".jpg")
    ])
    while len(files) >= CAPTURE_LIMIT:
        oldest = files.pop(0)
        try:
            os.remove(os.path.join(CAPTURE_DIR, oldest))
        except:
            pass
    with capture_index_lock:
        idx = capture_index
        capture_index += 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = str(idx).zfill(5) + "_" + ts + ".jpg"
    dest = os.path.join(CAPTURE_DIR, filename)
    shutil.copy2(IMAGE_PATH, dest)
    return filename


def play_sound_file(soundfile):
    sound_path = os.path.join(SOUND_DIR, soundfile)
    if not os.path.exists(sound_path):
        return False
    eventlet.tpool.execute(subprocess.run, ["play-audio", sound_path])
    return True


def loop_worker():
    global loop_running
    while True:
        with loop_lock:
            if not loop_running:
                break
            cfg = dict(loop_config)
            cfg["questions"] = list(loop_config["questions"])

        socketio.emit("loop_tick", {"phase": "capture_start"})

        cycle_start = time.time()

        ok = take_photo()
        if not ok:
            socketio.emit("loop_tick", {"phase": "error", "message": "Kamera nicht erreichbar"})
            eventlet.sleep(cfg["interval"])
            continue

        eventlet.sleep(0.5)

        loop_stats["total"] += 1
        loop_stats["last_capture"] = datetime.now().isoformat()

        if cfg["save_all"]:
            eventlet.tpool.execute(save_capture_file)

        socketio.emit("loop_tick", {
            "phase": "capture_done",
            "image_url": "/image?t=" + str(int(time.time() * 1000)),
            "stats": dict(loop_stats),
        })

        # Nächstes Foto parallel aufnehmen während VQA läuft
        next_result = [False]
        def _take_next():
            next_result[0] = take_photo_to(IMAGE_PATH_NEXT)
        next_thread = threading.Thread(target=_take_next, daemon=True)
        next_thread.start()

        results = []
        any_match = False
        for item in cfg["questions"]:
            question = item.get("question", "").strip()
            match_word = item.get("match", "").strip()
            cmd = item.get("cmd", "").strip()
            if not question:
                continue
            try:
                answer = run_vqa(question)
            except Exception as e:
                answer = "Fehler: " + str(e)
            matched = bool(match_word) and (match_word in answer)
            entry = {"question": question, "answer": answer, "matched": matched}
            results.append(entry)

            if matched:
                any_match = True
                loop_stats["alarms"] += 1
                if cfg["save"]:
                    saved = save_alarm_file(question, answer)
                    if saved:
                        loop_stats["saved"] += 1
                        entry["saved_file"] = saved
                if cfg["sound"]:
                    play_sound_file(cfg["soundfile"])
                if cmd:
                    try:
                        eventlet.tpool.execute(subprocess.run, cmd, shell=True)
                    except Exception as e:
                        entry["cmd_error"] = str(e)

            socketio.emit("loop_answer", {
                "result": entry,
                "stats": dict(loop_stats),
            })

        socketio.emit("loop_tick", {
            "phase": "cycle_done",
            "any_match": any_match,
            "results": results,
            "stats": dict(loop_stats),
        })

        # Warten bis nächstes Foto fertig, dann als aktuelles setzen
        next_thread.join()
        if next_result[0] and os.path.exists(IMAGE_PATH_NEXT):
            eventlet.sleep(0.3)
            shutil.move(IMAGE_PATH_NEXT, IMAGE_PATH)

        # Restzeit schlafen sodass Gesamtzyklus = interval
        elapsed = time.time() - cycle_start
        rest = cfg["interval"] - elapsed
        slept = 0.0
        step = 0.2
        while slept < rest:
            eventlet.sleep(step)
            slept += step
            with loop_lock:
                if not loop_running:
                    break

    socketio.emit("loop_state", {"running": False, "stats": dict(loop_stats)})


@app.route("/")
def index():
    if not os.path.exists(HTML_PATH):
        return "HTML nicht gefunden: " + HTML_PATH, 404
    return send_file(HTML_PATH, mimetype="text/html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "loop_running": loop_running})


@app.route("/photo", methods=["POST"])
def photo():
    if not take_photo():
        return jsonify({"error": "Kamera nicht erreichbar"}), 500
    return jsonify({"status": "ok"})


@app.route("/image", methods=["GET"])
def image():
    if not os.path.exists(IMAGE_PATH):
        return jsonify({"error": "Kein Bild vorhanden"}), 404
    return send_file(IMAGE_PATH, mimetype="image/jpeg")


@app.route("/save_alarm", methods=["POST"])
def save_alarm():
    data = request.get_json() or {}
    question = data.get("question", "")
    answer = data.get("answer", "")
    filename = save_alarm_file(question, answer)
    if not filename:
        return jsonify({"error": "Kein Bild vorhanden"}), 404
    return jsonify({"status": "ok", "file": filename})


@app.route("/alarms/<filename>", methods=["GET"])
def get_alarm(filename):
    path = os.path.join(ALARM_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Nicht gefunden"}), 404
    if filename.endswith(".json"):
        return send_file(path, mimetype="application/json")
    return send_file(path, mimetype="image/jpeg")


@app.route("/alarms/<filename>", methods=["DELETE"])
def delete_alarm(filename):
    path = os.path.join(ALARM_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Nicht gefunden"}), 404
    os.remove(path)
    json_path = path.replace(".jpg", ".json")
    if os.path.exists(json_path):
        os.remove(json_path)
    return jsonify({"status": "ok"})


@app.route("/play_sound", methods=["POST"])
def play_sound():
    data = request.get_json() or {}
    soundfile = data.get("file", "default_alarm_sound.wav")
    if not play_sound_file(soundfile):
        return jsonify({"error": "Datei nicht gefunden: " + soundfile}), 404
    return jsonify({"status": "ok"})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "question erforderlich"}), 400
    try:
        if "image" in data:
            image_bytes = base64.b64decode(data["image"])
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif "alarm_file" in data:
            fname = os.path.basename(data["alarm_file"])
            path = os.path.join(ALARM_DIR, fname)
            if not os.path.exists(path):
                return jsonify({"error": "Datei nicht gefunden"}), 404
            img = Image.open(path).convert("RGB")
        elif "capture_file" in data:
            fname = os.path.basename(data["capture_file"])
            path = os.path.join(CAPTURE_DIR, fname)
            if not os.path.exists(path):
                return jsonify({"error": "Datei nicht gefunden"}), 404
            img = Image.open(path).convert("RGB")
        else:
            if not os.path.exists(IMAGE_PATH):
                return jsonify({"error": "Kein Bild vorhanden"}), 404
            img = Image.open(IMAGE_PATH).convert("RGB")

        question = data["question"]
        inputs = processor(img, question, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_id = outputs.logits.argmax(-1).item()
        answer = model.config.id2label[predicted_id]

        # Folgefrage in JSON speichern wenn alarm_file angegeben
        if "alarm_file" in data:
            fname = os.path.basename(data["alarm_file"])
            json_path = os.path.join(ALARM_DIR, fname.replace(".jpg", ".json"))
            if os.path.exists(json_path):
                try:
                    with open(json_path) as f:
                        meta = json.load(f)
                    if "followups" not in meta:
                        meta["followups"] = []
                    meta["followups"].append({
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().isoformat()
                    })
                    with open(json_path, "w") as f:
                        json.dump(meta, f)
                except Exception:
                    pass

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/alarms", methods=["GET"])
def list_alarms():
    if not os.path.exists(ALARM_DIR):
        return jsonify({"alarms": []})
    jpgs = sorted([
        f for f in os.listdir(ALARM_DIR)
        if f.lower().endswith(".jpg")
    ])
    result = []
    for jpg in jpgs:
        entry = {"file": jpg}
        json_path = os.path.join(ALARM_DIR, jpg.replace(".jpg", ".json"))
        if os.path.exists(json_path):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
                entry["question"] = meta.get("question", "")
                entry["answer"] = meta.get("answer", "")
                entry["timestamp"] = meta.get("timestamp", "")
                entry["followups"] = meta.get("followups", [])
            except:
                pass
        result.append(entry)
    return jsonify({"alarms": result})


@app.route("/captures", methods=["GET"])
def list_captures():
    if not os.path.exists(CAPTURE_DIR):
        return jsonify({"files": [], "total": 0, "limit": CAPTURE_LIMIT})
    files = sorted([
        f for f in os.listdir(CAPTURE_DIR)
        if f.lower().endswith(".jpg")
    ], reverse=True)
    return jsonify({"files": files, "total": len(files), "limit": CAPTURE_LIMIT})


@app.route("/captures/<filename>", methods=["GET"])
def get_capture(filename):
    path = os.path.join(CAPTURE_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Nicht gefunden"}), 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/run_cmd", methods=["POST"])
def run_cmd():
    data = request.get_json() or {}
    cmd = data.get("cmd", "").strip()
    if not cmd:
        return jsonify({"error": "cmd fehlt"}), 400
    try:
        eventlet.tpool.execute(subprocess.run, cmd, shell=True)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/loop/start", methods=["POST"])
def loop_start():
    global loop_thread, loop_running
    data = request.get_json() or {}

    questions = data.get("questions", [])
    if not isinstance(questions, list) or len(questions) == 0:
        return jsonify({"error": "questions (Liste) erforderlich"}), 400

    interval = int(data.get("interval", 5))
    if interval < 1:
        interval = 1

    with loop_lock:
        if loop_running:
            return jsonify({"error": "Loop laeuft bereits"}), 409

        loop_config["interval"] = interval
        loop_config["questions"] = questions
        loop_config["save"] = bool(data.get("save", True))
        loop_config["save_all"] = bool(data.get("save_all", False))
        loop_config["sound"] = bool(data.get("sound", True))
        loop_config["soundfile"] = data.get("soundfile", "default_alarm_sound.wav")

        loop_stats["total"] = 0
        loop_stats["alarms"] = 0
        loop_stats["saved"] = 0
        loop_stats["last_capture"] = None
        loop_stats["started_at"] = datetime.now().isoformat()

        loop_running = True
        loop_thread = socketio.start_background_task(loop_worker)

    socketio.emit("loop_state", {"running": True, "config": dict(loop_config), "stats": dict(loop_stats)})
    return jsonify({"status": "ok", "config": loop_config, "stats": loop_stats})


@app.route("/loop/stop", methods=["POST"])
def loop_stop():
    global loop_running
    with loop_lock:
        loop_running = False
    return jsonify({"status": "ok"})


@app.route("/loop/status", methods=["GET"])
def loop_status():
    with loop_lock:
        return jsonify({
            "running": loop_running,
            "config": dict(loop_config),
            "stats": dict(loop_stats),
        })


@socketio.on("connect")
def on_connect():
    emit("loop_state", {
        "running": loop_running,
        "config": dict(loop_config),
        "stats": dict(loop_stats),
    })


if __name__ == "__main__":
    model_id  = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    model_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODELDIR
    load_model(model_id, model_dir)
    print("Server laeuft auf http://" + HOST + ":" + str(PORT))
    socketio.run(app, host=HOST, port=PORT)
