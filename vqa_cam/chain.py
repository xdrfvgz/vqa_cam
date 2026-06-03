#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_chain.py – rule chain evaluation

import re
import subprocess
import time
from vqa_cam.models import run_vqa, init_model
from vqa_cam.storage import save_alarm

RED   = "\033[91m"
GREEN = "\033[92m"
GRAY  = "\033[90m"
RESET = "\033[0m"


def match_answer(expr, answer):
    """Evaluate match expression against an answer string.

    Operators: >N  <N  >=N  <=N  !=N  (numeric, uses first number in answer)
               !word                   (string negation)
               word                    (substring, default)
    """
    if not expr:
        return False
    expr = expr.strip()
    m = re.match(r'^(>=|<=|!=|>|<)\s*(\d+(?:\.\d+)?)$', expr)
    if m:
        op, num = m.group(1), float(m.group(2))
        nums = re.findall(r'\d+(?:\.\d+)?', answer)
        if not nums:
            return op == '!='
        val = float(nums[0])
        return {'>': val > num, '<': val < num, '>=': val >= num,
                '<=': val <= num, '!=': val != num}[op]
    if expr.startswith('!'):
        word = expr[1:].strip()
        return bool(word) and word.lower() not in answer.lower()
    return expr.lower() in answer.lower()


def evaluate_chain(item, image_path, cfg, depth=0, chain_so_far=None, timg=False, quiet=False):
    if chain_so_far is None:
        chain_so_far = []

    question   = item.get("question", "").strip()
    match_word = item.get("match", "").strip()
    cmd        = item.get("cmd", "").strip()
    model      = item.get("model") or cfg.get("default_model") or None
    followup   = item.get("followup")
    is_leaf    = followup is None

    if not question:
        return []

    t0 = time.time()
    if model:
        init_model(model, quiet=quiet)

    indent = "  " * depth
    if not quiet:
        print(indent + GRAY + question + RESET + " ", end="", flush=True)

    try:
        answer = run_vqa(image_path, question, model)
    except Exception as e:
        answer = "error: " + str(e)

    elapsed = time.time() - t0
    matched = match_answer(match_word, answer)

    if not quiet:
        print((RED if matched else GREEN) + answer + RESET +
              GRAY + "  [{:.1f}s]".format(elapsed) + RESET)

    current_chain = chain_so_far + [{"question": question, "answer": answer, "matched": matched}]
    results = [{"question": question, "answer": answer, "matched": matched,
                "depth": depth, "is_leaf": is_leaf}]

    if matched:
        if is_leaf:
            if cfg.get("save"):
                saved = save_alarm(cfg, image_path, current_chain)
                if not quiet:
                    print(indent + GRAY + "saved: " + saved + RESET)
            sound = cfg.get("soundfile", "")
            if sound and __import__("os").path.exists(str(sound)):
                subprocess.Popen(["play-audio", sound])
        if cmd:
            print("\n\033[90mCommand: " + cmd + "\033[0m")
            subprocess.run(cmd, shell=True)
        if followup:
            results.extend(evaluate_chain(
                followup, image_path, cfg, depth + 1, current_chain, timg, quiet
            ))

    return results


def load_questions(args, cfg=None):
    import json
    if hasattr(args, "config") and args.config:
        with open(args.config) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "questions" in data:
            return data["questions"]
        return [data]
    if hasattr(args, "question") and args.question:
        return [{"question": " ".join(args.question),
                 "match": getattr(args, "match", "") or "",
                 "cmd":   getattr(args, "cmd",   "") or ""}]
    if cfg and cfg.get("questions"):
        return cfg["questions"]
    return []
