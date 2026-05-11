#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vqa_cli.py
#
# Usage:
#   python3 vqa_cli.py "Do you see a human?" "yes" --image foto.jpg
#   python3 vqa_cli.py "Do you see a human?" "yes" --image foto.jpg --cmd "termux-notification -t Alarm"
#   python3 vqa_cli.py "Do you see a human?" "yes" --image foto.jpg --model-dir ~/vilt-vqa
#
# Exit codes:
#   0 = match
#   1 = no match
#   2 = error

import sys
import os
import argparse
import subprocess
import contextlib
import io

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


def load_model(model_id, model_dir):
    from transformers import ViltProcessor, ViltForQuestionAnswering, logging
    logging.set_verbosity_error()
    import torch

    if not os.path.exists(model_dir):
        print("Downloading model: " + model_id, file=sys.stderr)
        processor = ViltProcessor.from_pretrained(model_id)
        model = ViltForQuestionAnswering.from_pretrained(model_id)
        processor.save_pretrained(model_dir)
        model.save_pretrained(model_dir, safe_serialization=True)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            processor = ViltProcessor.from_pretrained(model_dir, local_files_only=True)
            model = ViltForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    return processor, model


def run_vqa(processor, model, image_path, question):
    import torch
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    inputs = processor(img, question, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = outputs.logits.argmax(-1).item()
    return model.config.id2label[predicted_id]


def main():
    parser = argparse.ArgumentParser(
        description="VQA single-shot CLI – returns true/false based on model answer"
    )
    parser.add_argument("question", help="Question to ask about the image")
    parser.add_argument("match", help="Word that must appear in the answer to trigger a match")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--cmd", default="", help="Shell command to run on match")
    parser.add_argument("--model-id", default="dandelin/vilt-b32-finetuned-vqa",
                        help="HuggingFace model ID")
    parser.add_argument("--model-dir", default=os.path.expanduser("~/vilt-vqa"),
                        help="Local model directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress output, only use exit codes")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print("error: image not found: " + args.image, file=sys.stderr)
        sys.exit(2)

    try:
        processor, model = load_model(args.model_id, args.model_dir)
        answer = run_vqa(processor, model, args.image, args.question)
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


if __name__ == "__main__":
    main()
