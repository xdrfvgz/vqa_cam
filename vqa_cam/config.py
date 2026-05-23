#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_config.py – configuration defaults and loading

import os
import json

CONFIG_PATH = os.path.expanduser("~/vqa-scripts/config/vqa-ai-cam.json")

CAMERA_COMMANDS = {
    "termux":   "termux-camera-photo -c 2 {output}",
    "rpi":      "libcamera-still -o {output} --nopreview -t 1",
    "fswebcam": "fswebcam -r 1280x720 --no-banner {output}",
}

_MODEL_DIRS = {
    "vilt":   "~/vqa-models/vilt",
    "blip":   "~/vqa-models/blip",
    "blip-l": "~/vqa-models/blip-l",
    "git":    "~/vqa-models/git",
    "vbert":  "~/vqa-models/vbert",
}

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CLIENT_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "client")

DEFAULTS = {
    "model":         "blip",
    "model_dir":     None,
    "image_path":    "~/vqa-scripts/images/foto.jpg",
    "alarm_dir":     "~/vqa-scripts/alarms",
    "capture_dir":   "~/vqa-scripts/captures",
    "sound_dir":     "~/vqa-scripts/sounds",
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
