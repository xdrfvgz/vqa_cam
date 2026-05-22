#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_camera.py – camera and image utilities

import os
import subprocess

from vqa_cam.config import CAMERA_COMMANDS

RED   = "\033[91m"
GRAY  = "\033[90m"
RESET = "\033[0m"


def resolve_camera(camera):
    cmd = CAMERA_COMMANDS.get(camera, camera)
    if "{output}" not in cmd:
        print(RED + "Error: camera command must contain {output}" + RESET)
        import sys; sys.exit(2)
    return cmd


def take_photo(camera_cmd, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    before = os.path.getmtime(path) if os.path.exists(path) else 0
    result = subprocess.run(
        camera_cmd.replace("{output}", path),
        shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
        return False
    after = os.path.getmtime(path) if os.path.exists(path) else 0
    return after > before


def show_image(path, timg=False):
    if timg and os.path.exists(path):
        subprocess.run(["timg", path])
