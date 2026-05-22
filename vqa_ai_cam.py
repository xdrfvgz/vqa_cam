#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_ai_cam.py
#
# VQA-Cam – visual question answering for surveillance and automation
#
# Modes:
#   ask      – single-shot Q&A, exit 0=match, 1=no match (scriptable)
#   single   – analyze image interactively with followup questions
#   loop     – continuous capture with rule chain, alarms, commands
#   gallery  – browse saved alarm images in terminal
#   server   – start Flask/Socket.IO web server with browser UI
#   config   – show or init config file
#
# Config file: ~/vqa-scripts/config/vqa-ai-cam.json
#
# Usage:
#   python3 vqa_ai_cam.py ask "Do you see a human?" yes --image foto.jpg
#   python3 vqa_ai_cam.py single --camera termux --timg
#   python3 vqa_ai_cam.py loop --camera termux --interval 10 --config questions.json --save
#   python3 vqa_ai_cam.py gallery --timg --interactive
#   python3 vqa_ai_cam.py server
#   python3 vqa_ai_cam.py server --port 5666 --model blip
#   python3 vqa_ai_cam.py config init
#   python3 vqa_ai_cam.py config show

import sys
import os
import argparse
import subprocess
import threading
import shutil
import time
import json
import signal
from datetime import datetime

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from vqa_cam import config as vqa_config
from vqa_cam import models as vqa_models
from vqa_cam import camera as vqa_camera
from vqa_cam import storage as vqa_storage
from vqa_cam import chain as vqa_chain

# ── ANSI ─────────────────────────────────────────────────────────────────────

RED   = "\033[91m"
GREEN = "\033[92m"
BLUE  = "\033[94m"
GRAY  = "\033[90m"
BOLD  = "\033[1m"
RESET = "\033[0m"


# ── CLI Modes ─────────────────────────────────────────────────────────────────

def mode_ask(args, cfg):
    if not os.path.exists(args.image):
        print("error: image not found: " + args.image, file=sys.stderr)
        sys.exit(2)
    try:
        vqa_models.init_model(cfg["model"], cfg["model_dir"], quiet=args.quiet)
        answer = vqa_models.run_vqa(args.image, args.question)
    except Exception as e:
        print("error: " + str(e), file=sys.stderr)
        sys.exit(2)
    matched = args.match.strip().lower() in answer.strip().lower()
    if not args.quiet:
        print(answer)
        print("true" if matched else "false")
    if matched and args.cmd:
        subprocess.run(args.cmd, shell=True)
    sys.exit(0 if matched else 1)


def mode_single(args, cfg):
    image_path = args.image
    if not image_path:
        camera_cmd = vqa_camera.resolve_camera(args.camera)
        image_path = cfg["image_path"]
        print(GRAY + "Taking photo..." + RESET)
        if not vqa_camera.take_photo(camera_cmd, image_path):
            print(RED + "Error: could not take photo" + RESET)
            sys.exit(2)
    if not os.path.exists(image_path):
        print(RED + "Error: image not found: " + image_path + RESET)
        sys.exit(2)
    vqa_models.init_model(cfg["model"], cfg["model_dir"])
    vqa_camera.show_image(image_path, args.timg)
    questions = vqa_chain.load_questions(args)
    if questions:
        chain_cfg = {"save": args.save, "soundfile": "",
                     "alarm_dir": cfg["alarm_dir"], "capture_limit": cfg["capture_limit"]}
        for item in questions:
            vqa_chain.evaluate_chain(item, image_path, chain_cfg, timg=args.timg)
    print("\n" + GRAY + "Ask followup questions (empty to quit):" + RESET)
    while True:
        try:
            q = input(BLUE + "? " + RESET).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        print(GRAY + q + RESET + " ", end="", flush=True)
        try:
            print(GREEN + vqa_models.run_vqa(image_path, q) + RESET)
        except Exception as e:
            print(RED + "Error: " + str(e) + RESET)


def mode_loop(args, cfg):
    camera_cmd = vqa_camera.resolve_camera(args.camera)
    questions  = vqa_chain.load_questions(args)
    if not questions:
        print(RED + "Error: provide --config or --question" + RESET)
        sys.exit(2)

    vqa_models.init_model(cfg["model"], cfg["model_dir"])

    chain_cfg = {
        "save":          args.save,
        "soundfile":     args.sound or "",
        "save_all":      args.save_all,
        "alarm_dir":     cfg["alarm_dir"],
        "capture_dir":   cfg["capture_dir"],
        "capture_limit": cfg["capture_limit"],
    }

    image_path      = cfg["image_path"]
    image_path_next = image_path.replace(".jpg", "_next.jpg")
    running = True
    total   = 0
    alarms  = 0

    def handle_exit(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    print(BOLD + "VQA-Cam Loop" + RESET + "  –  Ctrl+C to stop")
    print(GRAY + "Camera  : " + camera_cmd + RESET)
    print(GRAY + "Interval: " + str(args.interval) + "s" + RESET)
    print(GRAY + "Model   : " + cfg["model"] + RESET)
    print()

    print(GRAY + "Taking first photo..." + RESET)
    if not vqa_camera.take_photo(camera_cmd, image_path):
        print(RED + "Error: could not take photo" + RESET)
        sys.exit(2)

    while running:
        cycle_start = time.time()
        total += 1
        ts = datetime.now().strftime("%H:%M:%S")
        print(BOLD + "[" + ts + "] #" + str(total) + RESET)

        vqa_camera.show_image(image_path, args.timg)

        if chain_cfg["save_all"]:
            vqa_storage.save_capture(chain_cfg, image_path)

        next_result = [False]
        def _take_next():
            next_result[0] = vqa_camera.take_photo(camera_cmd, image_path_next)
        next_thread = threading.Thread(target=_take_next, daemon=True)
        next_thread.start()

        for item in questions:
            results = vqa_chain.evaluate_chain(item, image_path, chain_cfg, timg=args.timg)
            if any(r.get("matched") and r.get("is_leaf") for r in results):
                alarms += 1

        next_thread.join()
        if next_result[0] and os.path.exists(image_path_next):
            time.sleep(0.3)
            shutil.move(image_path_next, image_path)

        elapsed = time.time() - cycle_start
        rest    = args.interval - elapsed
        if rest > 0 and running:
            time.sleep(rest)

    print("\n" + BOLD + "Stopped." + RESET +
          "  Total: " + str(total) + "  Alarms: " + str(alarms))


def mode_gallery(args, cfg):
    alarm_dir = cfg["alarm_dir"]
    alarms    = vqa_storage.load_alarms(alarm_dir)
    if not alarms:
        print(GRAY + "No alarms saved." + RESET)
        return

    files = sorted([a["file"] for a in alarms], reverse=True)
    alarm_map = {a["file"]: a for a in alarms}

    print(BOLD + "Alarm Gallery" + RESET + "  (" + str(len(files)) + " images)\n")

    for i, filename in enumerate(files):
        jpg      = os.path.join(alarm_dir, filename)
        meta     = alarm_map[filename]
        ts       = filename.replace(".jpg", "").replace("_", " ")
        print(BOLD + str(i + 1) + ". " + ts + RESET)
        chain = meta.get("chain", [])
        if chain:
            for j, c in enumerate(chain):
                color = RED if c.get("matched") else GREEN
                print("  " + "  " * j + GRAY + c["question"] + RESET + " → " + color + c["answer"] + RESET)
        elif meta.get("question"):
            print("  " + GRAY + meta["question"] + RESET + " → " + RED + meta["answer"] + RESET)
        for fu in meta.get("followups", []):
            print("  " + BLUE + fu["question"] + RESET + " → " + BLUE + fu["answer"] + RESET)

        vqa_camera.show_image(jpg, args.timg)
        print()

        if args.interactive:
            try:
                cmd = input(GRAY + "  [Enter=next  q=quit  d=delete] " + RESET).strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd == "q":
                break
            elif cmd == "d":
                os.remove(jpg)
                jpath = jpg.replace(".jpg", ".json")
                if os.path.exists(jpath):
                    os.remove(jpath)
                print(GRAY + "  deleted." + RESET)


def mode_config(args, cfg):
    if args.action == "init":
        vqa_config.init()
    elif args.action == "show":
        print(json.dumps(cfg, indent=2))


# ── Server mode ───────────────────────────────────────────────────────────────

def mode_server(args, cfg):
    import eventlet
    import eventlet.tpool
    from flask import Flask, request, jsonify, send_file
    from flask_cors import CORS
    from flask_socketio import SocketIO, emit

    IMAGE_PATH      = cfg["image_path"]
    IMAGE_PATH_NEXT = IMAGE_PATH.replace(".jpg", "_next.jpg")
    ALARM_DIR       = cfg["alarm_dir"]
    CAPTURE_DIR     = cfg["capture_dir"]
    SOUND_DIR       = cfg["sound_dir"]
    HTML_PATH       = cfg["html_path"]
    HOST            = cfg["host"]
    PORT            = cfg["port"]
    CAP_LIMIT       = cfg["capture_limit"]

    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)

    app      = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

    vqa_models.init_model(cfg["model"], cfg["model_dir"])

    loop_thread        = None
    loop_running       = False
    loop_lock          = threading.Lock()
    loop_config        = {"interval": 5, "questions": [], "save": True,
                          "save_all": False, "sound": True, "soundfile": "default_alarm_sound.wav"}
    loop_stats         = {"total": 0, "alarms": 0, "saved": 0,
                          "last_capture": None, "started_at": None}

    def srv_take_photo(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        before = os.path.getmtime(path) if os.path.exists(path) else 0
        result = subprocess.run(f"termux-camera-photo -c 2 {path}", shell=True)
        if result.returncode != 0:
            return False
        after = os.path.getmtime(path) if os.path.exists(path) else 0
        return after > before

    def srv_save_alarm(question, answer, chain=None):
        if not os.path.exists(IMAGE_PATH):
            return None
        alarm_cfg = {"alarm_dir": ALARM_DIR}
        full_chain = chain or [{"question": question, "answer": answer, "matched": True}]
        dest = eventlet.tpool.execute(vqa_storage.save_alarm, alarm_cfg, IMAGE_PATH, full_chain)
        return os.path.basename(dest) if dest else None

    def srv_save_capture():
        if not os.path.exists(IMAGE_PATH):
            return None
        cap_cfg = {"capture_dir": CAPTURE_DIR, "capture_limit": CAP_LIMIT}
        dest = eventlet.tpool.execute(vqa_storage.save_capture, cap_cfg, IMAGE_PATH)
        return os.path.basename(dest) if dest else None

    def srv_play_sound(soundfile):
        path = os.path.join(SOUND_DIR, soundfile)
        if not os.path.exists(path):
            return False
        eventlet.tpool.execute(subprocess.run, ["play-audio", path])
        return True

    def srv_evaluate_chain(item, srv_cfg, depth=0, chain_so_far=None):
        if chain_so_far is None:
            chain_so_far = []
        question   = item.get("question", "").strip()
        match_word = item.get("match", "").strip()
        cmd        = item.get("cmd", "").strip()
        followup   = item.get("followup")
        is_leaf    = followup is None
        if not question:
            return []
        try:
            answer = eventlet.tpool.execute(vqa_models.run_vqa, IMAGE_PATH, question)
        except Exception as e:
            answer = "Fehler: " + str(e)
        matched = bool(match_word) and (match_word in answer)
        entry   = {"question": question, "answer": answer, "matched": matched,
                   "depth": depth, "is_leaf": is_leaf}
        current_chain = chain_so_far + [{"question": question, "answer": answer, "matched": matched}]
        socketio.emit("loop_answer", {"result": entry, "stats": dict(loop_stats)})
        results = [entry]
        if matched:
            if is_leaf:
                loop_stats["alarms"] += 1
                if srv_cfg["save"]:
                    saved = srv_save_alarm(question, answer, chain=current_chain)
                    if saved:
                        loop_stats["saved"] += 1
                        entry["saved_file"] = saved
                if srv_cfg["sound"]:
                    srv_play_sound(srv_cfg["soundfile"])
            if cmd:
                try:
                    eventlet.tpool.execute(subprocess.run, cmd, shell=True)
                except Exception as e:
                    entry["cmd_error"] = str(e)
            if followup:
                results.extend(srv_evaluate_chain(followup, srv_cfg, depth + 1, current_chain))
        return results

    def loop_worker():
        nonlocal loop_running
        while True:
            with loop_lock:
                if not loop_running:
                    break
                srv_cfg = dict(loop_config)
                srv_cfg["questions"] = list(loop_config["questions"])
            socketio.emit("loop_tick", {"phase": "capture_start"})
            cycle_start = time.time()
            ok = eventlet.tpool.execute(srv_take_photo, IMAGE_PATH)
            if not ok:
                socketio.emit("loop_tick", {"phase": "error", "message": "Kamera nicht erreichbar"})
                eventlet.sleep(srv_cfg["interval"])
                continue
            eventlet.sleep(0.5)
            loop_stats["total"] += 1
            loop_stats["last_capture"] = datetime.now().isoformat()
            if srv_cfg["save_all"]:
                eventlet.tpool.execute(srv_save_capture)
            socketio.emit("loop_tick", {
                "phase": "capture_done",
                "image_url": "/image?t=" + str(int(time.time() * 1000)),
                "stats": dict(loop_stats),
            })
            next_result = [False]
            def _take_next():
                next_result[0] = eventlet.tpool.execute(srv_take_photo, IMAGE_PATH_NEXT)
            next_greenlet = eventlet.spawn(_take_next)
            results   = []
            any_match = False
            for item in srv_cfg["questions"]:
                chain_results = srv_evaluate_chain(item, srv_cfg)
                results.extend(chain_results)
                if any(r.get("matched") and r.get("is_leaf") for r in chain_results):
                    any_match = True
            next_greenlet.wait()
            if next_result[0] and os.path.exists(IMAGE_PATH_NEXT):
                eventlet.sleep(0.3)
                shutil.move(IMAGE_PATH_NEXT, IMAGE_PATH)
            socketio.emit("loop_tick", {"phase": "cycle_done", "any_match": any_match,
                                        "results": results, "stats": dict(loop_stats)})
            elapsed = time.time() - cycle_start
            rest    = srv_cfg["interval"] - elapsed
            slept   = 0.0
            while slept < rest:
                eventlet.sleep(0.2)
                slept += 0.2
                with loop_lock:
                    if not loop_running:
                        break
        socketio.emit("loop_state", {"running": False, "stats": dict(loop_stats)})

    @app.route("/")
    def index():
        if not os.path.exists(HTML_PATH):
            return "HTML not found: " + HTML_PATH, 404
        return send_file(HTML_PATH, mimetype="text/html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "loop_running": loop_running, "model": cfg["model"]})

    @app.route("/photo", methods=["POST"])
    def photo():
        if not eventlet.tpool.execute(srv_take_photo, IMAGE_PATH):
            return jsonify({"error": "Camera not available"}), 500
        return jsonify({"status": "ok"})

    @app.route("/image")
    def image():
        if not os.path.exists(IMAGE_PATH):
            return jsonify({"error": "No image"}), 404
        return send_file(IMAGE_PATH, mimetype="image/jpeg")

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "question required"}), 400
        requested_model = (data.get("model") or "").lower() or None
        if requested_model and requested_model not in vqa_models.loaded_models():
            try:
                eventlet.tpool.execute(vqa_models.init_model, requested_model, None, True)
            except Exception as e:
                return jsonify({"error": "failed to load model '{}': {}".format(requested_model, e)}), 500
        try:
            if "alarm_file" in data:
                path = os.path.join(ALARM_DIR, os.path.basename(data["alarm_file"]))
            elif "capture_file" in data:
                path = os.path.join(CAPTURE_DIR, os.path.basename(data["capture_file"]))
            else:
                path = IMAGE_PATH
            if not os.path.exists(path):
                return jsonify({"error": "File not found"}), 404
            answer = eventlet.tpool.execute(vqa_models.run_vqa, path, data["question"], requested_model)
            used_model = requested_model or vqa_models.current_model()
            if "alarm_file" in data:
                vqa_storage.add_followup(ALARM_DIR, os.path.basename(data["alarm_file"]),
                                         data["question"], answer)
            return jsonify({"answer": answer, "model": used_model})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/models")
    def list_models():
        return jsonify({
            "current":    vqa_models.current_model(),
            "loaded":     vqa_models.loaded_models(),
            "available":  list(vqa_models.MODEL_DEFAULTS.keys()),
            "max_loaded": vqa_models.MAX_LOADED,
        })

    @app.route("/save_alarm", methods=["POST"])
    def save_alarm_route():
        data     = request.get_json() or {}
        filename = srv_save_alarm(data.get("question", ""), data.get("answer", ""))
        if not filename:
            return jsonify({"error": "No image"}), 404
        return jsonify({"status": "ok", "file": filename})

    @app.route("/alarms")
    def list_alarms():
        return jsonify({"alarms": vqa_storage.load_alarms(ALARM_DIR)})

    @app.route("/alarms/<filename>")
    def get_alarm(filename):
        path = os.path.join(ALARM_DIR, filename)
        if not os.path.exists(path):
            return jsonify({"error": "Not found"}), 404
        return send_file(path, mimetype="application/json" if filename.endswith(".json") else "image/jpeg")

    @app.route("/alarms/<filename>", methods=["DELETE"])
    def delete_alarm(filename):
        path = os.path.join(ALARM_DIR, filename)
        if not os.path.exists(path):
            return jsonify({"error": "Not found"}), 404
        os.remove(path)
        jpath = path.replace(".jpg", ".json")
        if os.path.exists(jpath):
            os.remove(jpath)
        return jsonify({"status": "ok"})

    @app.route("/captures")
    def list_captures():
        if not os.path.exists(CAPTURE_DIR):
            return jsonify({"files": [], "total": 0, "limit": CAP_LIMIT})
        files = sorted([f for f in os.listdir(CAPTURE_DIR) if f.endswith(".jpg")], reverse=True)
        return jsonify({"files": files, "total": len(files), "limit": CAP_LIMIT})

    @app.route("/captures/<filename>")
    def get_capture(filename):
        path = os.path.join(CAPTURE_DIR, filename)
        if not os.path.exists(path):
            return jsonify({"error": "Not found"}), 404
        return send_file(path, mimetype="image/jpeg")

    @app.route("/play_sound", methods=["POST"])
    def play_sound():
        data = request.get_json() or {}
        if not srv_play_sound(data.get("file", "default_alarm_sound.wav")):
            return jsonify({"error": "File not found"}), 404
        return jsonify({"status": "ok"})

    @app.route("/run_cmd", methods=["POST"])
    def run_cmd():
        data = request.get_json() or {}
        cmd  = data.get("cmd", "").strip()
        if not cmd:
            return jsonify({"error": "cmd missing"}), 400
        eventlet.tpool.execute(subprocess.run, cmd, shell=True)
        return jsonify({"status": "ok"})

    @app.route("/loop/start", methods=["POST"])
    def loop_start():
        nonlocal loop_thread, loop_running
        data      = request.get_json() or {}
        questions = data.get("questions", [])
        if not questions:
            return jsonify({"error": "questions required"}), 400
        interval = max(1, int(data.get("interval", 5)))
        with loop_lock:
            if loop_running:
                return jsonify({"error": "Loop already running"}), 409
            loop_config.update({
                "interval":  interval,
                "questions": questions,
                "save":      bool(data.get("save", True)),
                "save_all":  bool(data.get("save_all", False)),
                "sound":     bool(data.get("sound", True)),
                "soundfile": data.get("soundfile", "default_alarm_sound.wav"),
            })
            loop_stats.update({"total": 0, "alarms": 0, "saved": 0,
                                "last_capture": None, "started_at": datetime.now().isoformat()})
            loop_running = True
            loop_thread  = socketio.start_background_task(loop_worker)
        socketio.emit("loop_state", {"running": True, "config": dict(loop_config), "stats": dict(loop_stats)})
        return jsonify({"status": "ok", "config": loop_config, "stats": loop_stats})

    @app.route("/loop/stop", methods=["POST"])
    def loop_stop():
        nonlocal loop_running
        with loop_lock:
            loop_running = False
        return jsonify({"status": "ok"})

    @app.route("/loop/status")
    def loop_status():
        with loop_lock:
            return jsonify({"running": loop_running, "config": dict(loop_config), "stats": dict(loop_stats)})

    @socketio.on("connect")
    def on_connect():
        emit("loop_state", {"running": loop_running, "config": dict(loop_config), "stats": dict(loop_stats)})

    print("Server running on http://" + HOST + ":" + str(PORT))
    socketio.run(app, host=HOST, port=PORT)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="vqa-ai-cam",
        description="VQA-Cam – visual question answering for surveillance and automation"
    )
    parser.add_argument("--alarm-dir",   default=None, help="Override alarm directory")
    parser.add_argument("--capture-dir", default=None, help="Override capture directory")
    parser.add_argument("--image-path",  default=None, help="Override current image path")

    sub = parser.add_subparsers(dest="mode", required=True)

    def add_model(p):
        p.add_argument("--model", default=None,
                       choices=["vilt", "blip", "blip-l", "git", "vbert"],
                       help="VQA model")
        p.add_argument("--model-dir", default=None, help="Local model cache directory")

    def add_camera(p):
        p.add_argument("--camera", default="termux",
                       help="Camera preset (termux|rpi|fswebcam) or custom with {output}")

    def add_timg(p):
        p.add_argument("--timg", action="store_true", help="Display images with timg")

    def add_questions(p):
        p.add_argument("--question", nargs="+", help="Question to ask")
        p.add_argument("--match",    default="",   help="Match word")
        p.add_argument("--cmd",      default="",   help="Shell command on match")
        p.add_argument("--config",   default=None, help="JSON question chain config")

    # ask
    p_ask = sub.add_parser("ask", help="Single-shot Q&A (exit 0=match, 1=no match)")
    p_ask.add_argument("question", help="Question")
    p_ask.add_argument("match",    help="Match word")
    p_ask.add_argument("--image",  required=True, help="Image file")
    p_ask.add_argument("--cmd",    default="",    help="Shell command on match")
    p_ask.add_argument("--quiet",  action="store_true")
    add_model(p_ask)

    # single
    p_single = sub.add_parser("single", help="Analyze one image interactively")
    add_model(p_single)
    add_camera(p_single)
    add_timg(p_single)
    add_questions(p_single)
    p_single.add_argument("--image", default=None, help="Image file (default: take photo)")
    p_single.add_argument("--save",  action="store_true", help="Save alarm images")

    # loop
    p_loop = sub.add_parser("loop", help="Continuous capture and analysis")
    add_model(p_loop)
    add_camera(p_loop)
    add_timg(p_loop)
    add_questions(p_loop)
    p_loop.add_argument("--interval", type=float, default=10, help="Cycle time in seconds")
    p_loop.add_argument("--save",     action="store_true", help="Save alarm images")
    p_loop.add_argument("--save-all", action="store_true", help="Save all captures")
    p_loop.add_argument("--sound",    default=None, help="Sound file on alarm")

    # gallery
    p_gallery = sub.add_parser("gallery", help="Browse alarm images in terminal")
    add_timg(p_gallery)
    p_gallery.add_argument("--interactive", action="store_true", help="Interactive mode")

    # server
    p_server = sub.add_parser("server", help="Start web server with browser UI")
    add_model(p_server)
    p_server.add_argument("--host", default=None, help="Host (default: 127.0.0.1)")
    p_server.add_argument("--port", type=int, default=None, help="Port (default: 5666)")
    p_server.add_argument("--html", default=None, help="Path to vqa_cam.html")

    # config
    p_cfg = sub.add_parser("config", help="Manage config file")
    p_cfg.add_argument("action", choices=["show", "init"], help="show or init")

    args = parser.parse_args()

    overrides = {}
    if hasattr(args, "alarm_dir")   and args.alarm_dir:   overrides["alarm_dir"]   = args.alarm_dir
    if hasattr(args, "capture_dir") and args.capture_dir: overrides["capture_dir"] = args.capture_dir
    if hasattr(args, "image_path")  and args.image_path:  overrides["image_path"]  = args.image_path
    if hasattr(args, "model")       and args.model:       overrides["model"]        = args.model
    if hasattr(args, "model_dir")   and args.model_dir:   overrides["model_dir"]   = args.model_dir
    if hasattr(args, "host")        and args.host:        overrides["host"]         = args.host
    if hasattr(args, "port")        and args.port:        overrides["port"]         = args.port
    if hasattr(args, "html")        and args.html:        overrides["html_path"]    = args.html

    cfg = vqa_config.load(overrides)

    if args.mode == "ask":
        mode_ask(args, cfg)
    elif args.mode == "single":
        mode_single(args, cfg)
    elif args.mode == "loop":
        mode_loop(args, cfg)
    elif args.mode == "gallery":
        mode_gallery(args, cfg)
    elif args.mode == "server":
        mode_server(args, cfg)
    elif args.mode == "config":
        mode_config(args, cfg)


if __name__ == "__main__":
    main()
