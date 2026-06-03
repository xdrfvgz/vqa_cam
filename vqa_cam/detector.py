#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# detector.py – pre-filter person detection (HOG and YOLO)


def detect(detector_type, image_path):
    if detector_type == "hog":
        return _hog(image_path)
    elif detector_type == "yolo":
        return _yolo(image_path)
    return {"count": 0, "detections": []}


def _hog(image_path):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found: " + image_path)
    s = 800 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img.copy()
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, weights = hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
    return {
        "count": len(boxes),
        "detections": [
            {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "confidence": float(weights[i])}
            for i, (x, y, w, h) in enumerate(boxes)
        ],
    }


def _yolo(image_path):
    import cv2
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found: " + image_path)
    results = model(img, verbose=False)
    persons = [
        float(box.conf[0])
        for box in results[0].boxes
        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5
    ]
    return {"count": len(persons), "detections": [{"confidence": k} for k in persons]}
