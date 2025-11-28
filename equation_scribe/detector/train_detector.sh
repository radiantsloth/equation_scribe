#!/usr/bin/env bash
# Quick training wrapper for YOLOv8 (ultralytics). Adjust epochs, imgsz, batch, and device.
set -e

MODEL=${1:-yolov8s.pt}
DATA_YAML=${2:-detector/detector.yaml}
EPOCHS=${3:-50}
IMGSZ=${4:-1024}
BATCH=${5:-8}
NAME=${6:-eq_detector_yolov8s}

echo "Training YOLOv8: model=${MODEL} data=${DATA_YAML} epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH}"
yolo task=detect mode=train model=${MODEL} data=${DATA_YAML} epochs=${EPOCHS} imgsz=${IMGSZ} batch=${BATCH} device=0 name=${NAME}
