#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_config.py – configuration defaults and loading

import os
import json
import shutil

CONFIG_PATH = os.path.expanduser("~/vqa_ai_cam/config/vqa-ai-cam.json")

CAMERA_COMMANDS = {
    "termux":   "termux-camera-photo -c 2 {output}",
    "rpi":      "libcamera-still -o {output} --nopreview -t 1",
    "fswebcam": "fswebcam -r 1280x720 --no-banner {output}",
}

_MODEL_DIRS = {
    "vilt":      "~/vqa-models/vilt",
    "blip":      "~/vqa-models/blip",
    "blip-l":    "~/vqa-models/blip-l",
    "git":       "~/vqa-models/git",
    "vbert":     "~/vqa-models/vbert",
    "moondream": "~/vqa-models/moondream",
    "blip2":     "~/vqa-models/blip2",
    "phi3v":     "~/vqa-models/phi3v",
    "llava":      "~/vqa-models/llava",
    "phi3v-onnx":  "~/vqa-models/phi3v-onnx",
}

_HF_MODEL_IDS = {
    "vilt":       "dandelin/vilt-b32-finetuned-vqa",
    "blip":       "Salesforce/blip-vqa-base",
    "blip-l":     "Salesforce/blip-vqa-capfilt-large",
    "git":        "microsoft/git-base-vqav2",
    "vbert":      "uclanlp/visualbert-vqa",
    "moondream":  "vikhyatk/moondream2",
    "blip2":      "Salesforce/blip2-opt-2.7b",
    "phi3v":      "microsoft/Phi-3.5-vision-instruct",
    "llava":      "llava-hf/llava-1.5-7b-hf",
    "phi3v-onnx": "microsoft/Phi-3-vision-128k-instruct-onnx-cpu",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "client")

DEFAULTS = {
    "model":         "blip",
    "model_dir":     None,
    "image_path":    "~/vqa_ai_cam/images/foto.jpg",
    "alarm_dir":     "~/vqa_ai_cam/alarms",
    "capture_dir":   "~/vqa_ai_cam/captures",
    "sound_dir":     "~/vqa_ai_cam/sounds",
    "html_path":     os.path.join(_CLIENT_DIR, "vqa_cam.html"),
    "host":          "127.0.0.1",
    "port":          5666,
    "capture_limit": 100,
}

_PATH_KEYS = ("image_path", "alarm_dir", "capture_dir", "sound_dir", "html_path", "model_dir")


def load(overrides=None):
    cfg = dict(DEFAULTS)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                stored = json.load(f)
            if isinstance(stored, dict):
                cfg.update({k: v for k, v in stored.items() if v is not None})
        except Exception:
            pass
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    for key in _PATH_KEYS:
        if cfg.get(key):
            cfg[key] = os.path.expanduser(str(cfg[key]))
    if not cfg.get("model_dir"):
        model = cfg.get("model", "blip")
        cfg["model_dir"] = os.path.expanduser(_MODEL_DIRS.get(model, "~/vqa-models/" + model))
    return cfg


def edit():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if not os.path.exists(CONFIG_PATH):
        init()
    import subprocess
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    return subprocess.call([editor, CONFIG_PATH])


def validate():
    issues = []
    if not os.path.exists(CONFIG_PATH):
        return ["Config file does not exist: " + CONFIG_PATH + " (run 'config init')"]
    try:
        with open(CONFIG_PATH) as f:
            stored = json.load(f)
    except json.JSONDecodeError as e:
        return ["Invalid JSON: " + str(e)]
    if not isinstance(stored, dict):
        return ["Top-level must be a JSON object"]
    cfg = load()
    if cfg.get("model") not in _MODEL_DIRS:
        issues.append("Unknown model: " + str(cfg.get("model")) +
                      " (known: " + ", ".join(_MODEL_DIRS) + ")")
    for key in ("alarm_dir", "capture_dir", "sound_dir"):
        path = cfg.get(key)
        if path and not os.path.exists(path):
            issues.append(key + " does not exist (will be created on use): " + path)
    html = cfg.get("html_path")
    if html and not os.path.exists(html):
        issues.append("html_path does not exist: " + html)
    port = cfg.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        issues.append("Invalid port: " + str(port))
    questions = stored.get("questions", [])
    if questions and not isinstance(questions, list):
        issues.append("'questions' must be a list")
    else:
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                issues.append("questions[" + str(i) + "] must be an object")
                continue
            if not q.get("question"):
                issues.append("questions[" + str(i) + "] missing 'question'")
    return issues


def save(updates):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    stored = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                stored = loaded
        except Exception:
            pass
    stored.update({k: v for k, v in updates.items() if v is not None})
    with open(CONFIG_PATH, "w") as f:
        json.dump(stored, f, indent=2)
    return CONFIG_PATH


def purge_model(model, model_dir):
    removed = []
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        removed.append(model_dir)
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_id = _HF_MODEL_IDS.get(model)
    if model_id:
        hf_dir = os.path.join(hf_cache, "models--" + model_id.replace("/", "--"))
        if os.path.exists(hf_dir):
            shutil.rmtree(hf_dir)
            removed.append(hf_dir)
    tmp_count = 0
    if os.path.exists(hf_cache):
        for dirpath, _, filenames in os.walk(hf_cache):
            for fname in filenames:
                if fname.endswith(".incomplete") or fname.startswith("tmp_"):
                    try:
                        os.remove(os.path.join(dirpath, fname))
                        tmp_count += 1
                    except OSError:
                        pass
    return removed, tmp_count


def init():
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    if os.path.exists(CONFIG_PATH):
        print("Config already exists: " + CONFIG_PATH)
        print(json.dumps(load(), indent=2))
        return
    saveable = {k: v for k, v in DEFAULTS.items() if k not in ("html_path", "model_dir")}
    with open(CONFIG_PATH, "w") as f:
        json.dump(saveable, f, indent=2)
    print("Created: " + CONFIG_PATH)
    print(json.dumps(load(), indent=2))
