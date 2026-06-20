#!/usr/bin/env bash
# Day 3: train 400M difficulty verifier
set -e
mkdir -p outputs/verifier-400m
python src/adaptivethink/verifier/train.py \
  --train data/teacher_labels.jsonl \
  --eval  data/verifier_eval_labelled.jsonl \
  --out   outputs/verifier-400m/best.pt \
  --epochs 3 --batch 32 --lr 2e-5
