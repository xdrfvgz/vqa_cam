#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_models.py – model loading and inference
#
# Supported models:
#   vilt   – dandelin/vilt-b32-finetuned-vqa   (~200MB, fast, fixed vocab)
#   blip   – Salesforce/blip-vqa-base          (~400MB, generative)
#   blip-l – Salesforce/blip-vqa-capfilt-large (~900MB, better accuracy)
#   git    – microsoft/git-base-vqav2          (~700MB, generative)
#   vbert  – uclanlp/visualbert-vqa            (~400MB, fixed vocab, needs detector)

import os
import contextlib
import io

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

MODEL_DEFAULTS = {
    "vilt":   {"model_id": "dandelin/vilt-b32-finetuned-vqa",       "dir": "~/vqa-models/vilt"},
    "blip":   {"model_id": "Salesforce/blip-vqa-base",              "dir": "~/vqa-models/blip"},
    "blip-l": {"model_id": "Salesforce/blip-vqa-capfilt-large",     "dir": "~/vqa-models/blip-l"},
    "git":    {"model_id": "microsoft/git-base-vqav2",              "dir": "~/vqa-models/git"},
    "vbert":  {"model_id": "uclanlp/visualbert-vqa",                "dir": "~/vqa-models/vbert"},
}

_processor  = None
_model      = None
_model_name = None


def init_model(model_name, model_dir=None, quiet=False):
    global _processor, _model, _model_name
    model_name = model_name.lower()
    if model_name not in MODEL_DEFAULTS:
        raise ValueError("Unknown model: " + model_name + ". Choose: " + ", ".join(MODEL_DEFAULTS))
    defaults  = MODEL_DEFAULTS[model_name]
    model_id  = defaults["model_id"]
    model_dir = model_dir or os.path.expanduser(defaults["dir"])
    if not quiet:
        print("\033[90mLoading model: " + model_name + "\033[0m")
    if model_name == "vilt":
        _processor, _model = _load_vilt(model_id, model_dir)
    elif model_name in ("blip", "blip-l"):
        _processor, _model = _load_blip(model_id, model_dir)
    elif model_name == "git":
        _processor, _model = _load_git(model_id, model_dir)
    elif model_name == "vbert":
        _processor, _model = _load_vbert(model_id, model_dir)
    _model_name = model_name
    return _processor, _model, _model_name


def run_vqa(image_path, question):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    if _model_name == "vilt":
        return _run_vilt(_processor, _model, img, question)
    elif _model_name in ("blip", "blip-l"):
        return _run_blip(_processor, _model, img, question)
    elif _model_name == "git":
        return _run_git(_processor, _model, img, question)
    elif _model_name == "vbert":
        return _run_vbert(_processor, _model, img, question)
    raise ValueError("No model loaded")


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
    return _tpool(_infer)


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
