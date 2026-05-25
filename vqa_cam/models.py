#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_cam/models.py – model loading, LRU cache, and inference
#
# Supported models:
#   vilt      – dandelin/vilt-b32-finetuned-vqa       (~200MB, fast, fixed vocab)
#   blip      – Salesforce/blip-vqa-base              (~400MB, generative)
#   blip-l    – Salesforce/blip-vqa-capfilt-large     (~900MB, better accuracy)
#   git       – microsoft/git-base-vqav2              (~700MB, generative)
#   vbert     – uclanlp/visualbert-vqa                (~400MB, fixed vocab, needs detector)
#   moondream – vikhyatk/moondream2                   (~900MB int8, full sentences)
#   blip2     – Salesforce/blip2-opt-2.7b             (~5GB, much better than blip)
#   phi3v     – microsoft/Phi-3.5-vision-instruct     (~4.2GB, excellent reasoning)
#   llava     – llava-hf/llava-1.5-7b-hf              (~4GB 4-bit, needs bitsandbytes)
#   phi3v-onnx– microsoft/Phi-3-vision-128k-instruct-onnx-cpu (~4GB int4, CPU-only, no GPU needed)
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
    "vilt":      {"model_id": "dandelin/vilt-b32-finetuned-vqa",       "dir": "~/vqa-models/vilt"},
    "blip":      {"model_id": "Salesforce/blip-vqa-base",              "dir": "~/vqa-models/blip"},
    "blip-l":    {"model_id": "Salesforce/blip-vqa-capfilt-large",     "dir": "~/vqa-models/blip-l"},
    "git":       {"model_id": "microsoft/git-base-vqav2",              "dir": "~/vqa-models/git"},
    "vbert":     {"model_id": "uclanlp/visualbert-vqa",                "dir": "~/vqa-models/vbert"},
    "moondream": {"model_id": "vikhyatk/moondream2",                   "dir": "~/vqa-models/moondream"},
    "blip2":     {"model_id": "Salesforce/blip2-opt-2.7b",             "dir": "~/vqa-models/blip2"},
    "phi3v":     {"model_id": "microsoft/Phi-3.5-vision-instruct",     "dir": "~/vqa-models/phi3v"},
    "llava":     {"model_id": "llava-hf/llava-1.5-7b-hf",             "dir": "~/vqa-models/llava"},
    "phi3v-onnx": {"model_id": "microsoft/Phi-3-vision-128k-instruct-onnx-cpu",
                                                                       "dir": "~/vqa-models/phi3v-onnx"},
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
        elif model_name == "moondream":
            p, m = _load_moondream(model_id, model_dir)
        elif model_name == "blip2":
            p, m = _load_blip2(model_id, model_dir)
        elif model_name == "phi3v":
            p, m = _load_phi3v(model_id, model_dir)
        elif model_name == "llava":
            p, m = _load_llava(model_id, model_dir)
        elif model_name == "phi3v-onnx":
            p, m = _load_phi3v_onnx(model_id, model_dir)
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
    elif model_name == "moondream":
        return _run_moondream(processor, model, img, question)
    elif model_name == "blip2":
        return _run_blip2(processor, model, img, question)
    elif model_name == "phi3v":
        return _run_phi3v(processor, model, img, question)
    elif model_name == "llava":
        return _run_llava(processor, model, img, question)
    elif model_name == "phi3v-onnx":
        return _run_phi3v_onnx(processor, model, img, question)


def current_model():
    with _lock:
        if not _cache:
            return None
        return next(reversed(_cache))


def loaded_models():
    with _lock:
        return list(_cache.keys())


def _tpool(fn):
    import sys
    if "eventlet.tpool" in sys.modules:
        import eventlet.tpool
        return eventlet.tpool.execute(fn)
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


def _load_moondream(model_id, model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, logging as tlog
    tlog.set_verbosity_error()
    # Newer transformers calls self.all_tied_weights_keys but some versions don't define it
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = property(
            lambda self: {k: None for k in (getattr(self, '_tied_weights_keys', None) or [])}
        )
    has_weights = os.path.isdir(model_dir) and any(
        f.endswith(".safetensors") or f.endswith(".bin")
        for f in os.listdir(model_dir)
    )
    if not has_weights:
        print("Downloading Moondream2 (~3.9GB)...")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, local_dir=model_dir)
    with contextlib.redirect_stderr(io.StringIO()):
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=True,
            torch_dtype="auto",
        )
    m.eval()
    return tok, m


def _run_moondream(tokenizer, model, img, question):
    def _infer():
        enc = model.encode_image(img)
        return model.answer_question(enc, question, tokenizer)
    return _tpool(_infer)


def _load_blip2(model_id, model_dir):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration, logging as tlog
    tlog.set_verbosity_error()
    if not os.path.exists(model_dir):
        print("Downloading BLIP-2 (~5GB)...")
        p = Blip2Processor.from_pretrained(model_id)
        m = Blip2ForConditionalGeneration.from_pretrained(model_id)
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = Blip2Processor.from_pretrained(model_dir, local_files_only=True)
            m = Blip2ForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
    m.eval()
    return p, m


def _run_blip2(processor, model, img, question):
    import torch
    inputs = processor(images=img, text=question, return_tensors="pt")
    def _infer():
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    return _tpool(_infer)


def _load_phi3v(model_id, model_dir):
    from transformers import AutoModelForCausalLM, AutoProcessor, logging as tlog
    tlog.set_verbosity_error()
    kwargs = {"trust_remote_code": True, "num_crops": 4}
    if not os.path.exists(model_dir):
        print("Downloading Phi-3.5-Vision (~4.2GB)...")
        p = AutoProcessor.from_pretrained(model_id, **kwargs)
        m = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                  torch_dtype="auto", _attn_implementation="eager")
        p.save_pretrained(model_dir)
        m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = AutoProcessor.from_pretrained(model_dir, local_files_only=True, **kwargs)
            m = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                      trust_remote_code=True, torch_dtype="auto",
                                                      _attn_implementation="eager")
    m.eval()
    return p, m


def _run_phi3v(processor, model, img, question):
    import torch
    messages = [{"role": "user", "content": "<|image_1|>\n" + question}]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [img], return_tensors="pt")
    def _infer():
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=100,
                                 eos_token_id=processor.tokenizer.eos_token_id)
        ids = ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return _tpool(_infer)


def _load_llava(model_id, model_dir):
    from transformers import LlavaForConditionalGeneration, AutoProcessor, logging as tlog
    tlog.set_verbosity_error()
    try:
        import bitsandbytes  # noqa
        load_kwargs = {"load_in_4bit": True}
    except ImportError:
        print("\033[93mWarning: bitsandbytes not installed – LLaVA loads in fp32 (~14GB). "
              "Install with: pip install bitsandbytes\033[0m")
        load_kwargs = {}
    if not os.path.exists(model_dir):
        print("Downloading LLaVA-1.5-7B (~4GB 4-bit / ~14GB fp32)...")
        p = AutoProcessor.from_pretrained(model_id)
        m = LlavaForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
        p.save_pretrained(model_dir)
        if not load_kwargs:
            m.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            p = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            m = LlavaForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
                                                              **load_kwargs)
    m.eval()
    return p, m


def _run_llava(processor, model, img, question):
    import torch
    prompt = "USER: <image>\n" + question + "\nASSISTANT:"
    inputs = processor(text=prompt, images=img, return_tensors="pt")
    def _infer():
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=100)
        full = processor.decode(out[0], skip_special_tokens=True)
        return full.split("ASSISTANT:")[-1].strip()
    return _tpool(_infer)


def _find_onnx_dir(base_dir):
    if not os.path.isdir(base_dir):
        return None
    if os.path.exists(os.path.join(base_dir, "genai_config.json")):
        return base_dir
    for name in sorted(os.listdir(base_dir)):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "genai_config.json")):
            return candidate
    return None


def _load_phi3v_onnx(model_id, model_dir):
    try:
        import onnxruntime_genai as og  # noqa
    except ImportError:
        raise RuntimeError("pip install onnxruntime-genai")
    onnx_dir = _find_onnx_dir(model_dir)
    if onnx_dir is None:
        from huggingface_hub import snapshot_download
        print("Downloading Phi-3-Vision ONNX int4 (~4GB)...")
        snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            allow_patterns=["cpu-int4-rtn-block-32-acc-level-4/**", "*.md"],
        )
        onnx_dir = _find_onnx_dir(model_dir)
    if onnx_dir is None:
        raise RuntimeError("Kein ONNX-Modell in " + model_dir + " gefunden (genai_config.json fehlt)")
    import onnxruntime_genai as og
    model = og.Model(onnx_dir)
    processor = model.create_multimodal_processor()
    return processor, model


def _run_phi3v_onnx(processor, model, img, question):
    import onnxruntime_genai as og
    import tempfile

    def _infer():
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path, "JPEG")
            images = og.Images.open(tmp_path)
            prompt = "<|user|>\n<|image_1|>\n" + question + "<|end|>\n<|assistant|>\n"
            inputs = processor(prompt, images=images)
            params = og.GeneratorParams(model)
            params.set_inputs(inputs)
            params.set_search_options(max_length=512)
            generator = og.Generator(model, params)
            parts = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                parts.append(tokenizer_stream.decode(generator.get_next_tokens()[0]))
            return "".join(parts).strip()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return _tpool(_infer)
