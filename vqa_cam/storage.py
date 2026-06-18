#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_storage.py – alarm and capture storage

import os
import shutil
import json
from datetime import datetime

GRAY  = "\033[90m"
RESET = "\033[0m"


def save_alarm(cfg, image_path, chain):
    os.makedirs(cfg["alarm_dir"], exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(cfg["alarm_dir"], ts + ".jpg")
    shutil.copy2(image_path, dest)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "question":  chain[0]["question"] if chain else "",
        "answer":    chain[0]["answer"]   if chain else "",
        "chain":     chain,
    }
    with open(os.path.join(cfg["alarm_dir"], ts + ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    return dest


def save_capture(cfg, image_path):
    os.makedirs(cfg["capture_dir"], exist_ok=True)
    files = sorted([f for f in os.listdir(cfg["capture_dir"]) if f.endswith(".jpg")])
    while len(files) >= cfg["capture_limit"]:
        os.remove(os.path.join(cfg["capture_dir"], files.pop(0)))
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(cfg["capture_dir"], ts + ".jpg")
    shutil.copy2(image_path, dest)
    return dest


def load_alarms(alarm_dir):
    if not os.path.exists(alarm_dir):
        return []
    result = []
    for jpg in sorted([f for f in os.listdir(alarm_dir) if f.endswith(".jpg")]):
        entry = {"file": jpg}
        jpath = os.path.join(alarm_dir, jpg.replace(".jpg", ".json"))
        if os.path.exists(jpath):
            try:
                with open(jpath) as f:
                    meta = json.load(f)
                entry.update({
                    "question":  meta.get("question", ""),
                    "answer":    meta.get("answer", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "followups": meta.get("followups", []),
                    "chain":     meta.get("chain", []),
                })
            except:
                pass
        result.append(entry)
    return result


def add_followup(alarm_dir, filename, question, answer):
    jpath = os.path.join(alarm_dir, filename.replace(".jpg", ".json"))
    if not os.path.exists(jpath):
        return False
    try:
        with open(jpath) as f:
            meta = json.load(f)
        meta.setdefault("followups", []).append({
            "question":  question,
            "answer":    answer,
            "timestamp": datetime.now().isoformat(),
        })
        with open(jpath, "w") as f:
            json.dump(meta, f)
        return True
    except:
        return False
