#!/usr/bin/env bash
# Day 2b: materialize GRPO router training data.
# train_grpo._load_data reads {question, answer} rows and computes difficulty
# itself via the verifier, so this only needs GSM8K TRAIN {question, answer}.
set -e
mkdir -p data
python src/adaptivethink/data/loaders.py \
  --dump gsm8k train data/gsm8k_train_labelled.jsonl
