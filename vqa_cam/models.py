#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_cam/models.py – model loading, LRU cache, and inference
#
# Supported models:
#   vilt   – dandelin/vilt-b32-finetuned-vqa   (~200MB, fast, fixed vocab)
#   blip   – Salesforce/blip-vqa-base          (~400MB, generative)
#   blip-l – Salesforce/blip-vqa-capfilt-large (~900MB, better accuracy)
#   git    – microsoft/git-base-vqav2          (~700MB, generative)
#   vbert  – uclanlp/visualbert-vqa            (~400MB, fixed vocab, needs detector)
#
# Cache: up to MAX_LOADED models held in memory at once (LRU eviction).
# Configure via env VQA_MAX_LOADED_MODELS (default 2).

import os
import contextlib
import io
import threading
from collections import OrderedDict

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODEL_DEFAULTS = {
    "vilt":   {"model_id": "dandelin/vilt-b32-finetuned-vqa",       "dir": "~/vqa-models/vilt"},
    "blip":   {"model_id": "Salesforce/blip-vqa-base",              "dir": "~/vqa-models/blip"},
    "blip-l": {"model_id": "Salesforce/blip-vqa-capfilt-large",     "dir": "~/vqa-models/blip-l"},
    "git":    {"model_id": "microsoft/git-base-vqav2",              "dir": "~/vqa-models/git"},
    "vbert":  {"model_id": "uclanlp/visualbert-vqa",                "dir": "~/vqa-models/vbert"},
}

MAX_LOADED = max(1, int(os.environ.get("VQA_MAX_LOADED_MODELS", "2")))

_cache = OrderedDict()
_lock  = threading.RLock()


def init_model(model_name, model_dir=None, quiet=False):
    model_name = model_name.lower()
    if model_name not in MODEL_DEFAULTS:
        raise ValueError("Unknown model: " + model_name + ". Choose: " + ", ".join(MODEL_DEFAULTS))
    with _lock:
        if model_name in _cache:
            _cache.move_to_end(model_name)
            return model_name
        while len(_cache) >= MAX_LOADED:
            evicted = next(iter(_cache))
            del _cache[evicted]
            if not quiet:
                print("\033[90mEvicted from cache: " + evicted + "\033[0m")
        if not quiet:
            print("\033[90mLoading model: " + model_name + "\033[0m")
        defaults  = MODEL_DEFAULTS[model_name]
        model_id  = defaults["model_id"]
        model_dir = model_dir or os.path.expanduser(defaults["dir"])
        if model_name == "vilt":
            p, m = _load_vilt(model_id, model_dir)
        elif model_name in ("blip", "blip-l"):
            p, m = _load_blip(model_id, model_dir)
        elif model_name == "git":
            p, m = _load_git(model_id, model_dir)
        elif model_name == "vbert":
            p, m = _load_vbert(model_id, model_dir)
        _cache[model_name] = (p, m)
        return model_name


def run_vqa(image_path, question, model_name=None):
    from PIL import Image
    with _lock:
        if model_name:
            model_name = model_name.lower()
            if model_name not in _cache:
                raise ValueError("Model '" + model_name + "' not loaded")
            _cache.move_to_end(model_name)
        else:
            if not _cache:
                raise ValueError("No model loaded")
            model_name = next(reversed(_cache))
        processor, model = _cache[model_name]
    img = Image.open(image_path).convert("RGB")
    if model_name == "vilt":
        return _run_vilt(processor, model, img, question)
    elif model_name in ("blip", "blip-l"):
        return _run_blip(processor, model, img, question)
    elif model_name == "git":
        return _run_git(processor, model, img, question)
    elif model_name == "vbert":
        return _run_vbert(processor, model, img, question)


def current_model():
    with _lock:
        if not _cache:
            return None
        return next(reversed(_cache))


def loaded_models():
    with _lock:
        return list(_cache.keys())


def _tpool(fn):
    try:
        import eventlet.tpool
        return eventlet.tpool.execute(fn)
    except ImportError:
        return fn()


def _load_vilt(model_id, model_dir):
    from transformers import ViltProcessor, ViltForQuestionAnswering, logging as tlog
    tlog.set_verbosity_error()
    if not os.path.exists(model_dir):
        print("Downloading ViLT...")
        p = ViltProcessor.from_pretrained(model_id)
        m = ViltForQuestionAnswering.from_pretrained(model_id)
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = ViltProcessor.from_pretrained(model_dir, local_files_only=True)
            m = ViltForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    m.eval()
    return p, m


def _run_vilt(processor, model, img, question):
    import torch
    inputs = processor(img, question, return_tensors="pt")
    def _infer():
        with torch.no_grad():
            outputs = model(**inputs)
        pid = outputs.logits.argmax(-1).item()
        return model.config.id2label[pid]
    return _tpool(_infer)


def _load_blip(model_id, model_dir):
    from transformers import BlipProcessor, BlipForQuestionAnswering, logging as tlog
    tlog.set_verbosity_error()
    if not os.path.exists(model_dir):
        print("Downloading BLIP...")
        p = BlipProcessor.from_pretrained(model_id)
        m = BlipForQuestionAnswering.from_pretrained(model_id)
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
            m = BlipForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    m.eval()
    return p, m


def _run_blip(processor, model, img, question):
    import torch
    inputs = processor(img, question, return_tensors="pt")
    def _infer():
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20)
        return processor.decode(out[0], skip_special_tokens=True)
    return _tpool(_infer)


def _load_git(model_id, model_dir):
    from transformers import AutoProcessor, AutoModelForCausalLM, logging as tlog
    tlog.set_verbosity_error()
    if not os.path.exists(model_dir):
        print("Downloading GIT...")
        p = AutoProcessor.from_pretrained(model_id)
        m = AutoModelForCausalLM.from_pretrained(model_id)
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
    m.eval()
    return p, m


def _run_git(processor, model, img, question):
    import torch
    pv  = processor(images=img, return_tensors="pt").pixel_values
    ids = processor(text=question, return_tensors="pt").input_ids
    def _infer():
        with torch.no_grad():
            gen = model.generate(pixel_values=pv, input_ids=ids, max_new_tokens=20)
        return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return _tpool()


def _load_vbert(model_id, model_dir):
    print("\033[93mWarning: VisualBERT requires object-detector visual features. "
          "Dummy embeddings are used here, so answers will be unreliable.\033[0m")
    from transformers import BertTokenizer, VisualBertForQuestionAnswering, logging as tlog
    tlog.set_verbosity_error()
    if not os.path.exists(model_dir):
        print("Downloading VisualBERT...")
        p = BertTokenizer.from_pretrained("bert-base-uncased")
        m = VisualBertForQuestionAnswering.from_pretrained(model_id)
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
            m = VisualBertForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    m.eval()
    return p, m


def _run_vbert(tokenizer, model, img, question):
    import torch
    inputs = tokenizer(question, return_tensors="pt", padding=True)
    hidden_size = model.config.visual_embedding_dim if hasattr(model.config, "visual_embedding_dim") else 2048
    inputs.update({
        "visual_embeds":         torch.zeros((1, 1, hidden_size)),
        "visual_token_type_ids": torch.zeros((1, 1), dtype=torch.long),
        "visual_attention_mask": torch.ones((1, 1),  dtype=torch.long),
    })
    def _infer():
        with torch.no_grad():
            out = model(**inputs)
        pid = out.logits.argmax(-1).item()
        return model.config.id2label[pid]
    return _tpool(_infer)
