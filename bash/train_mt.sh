#!/bin/bash
#SBATCH --mem=30G
#SBATCH --time=1-0
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH -c10

CUDA_VISIBLE_DEVICES=0 \
allennlp train \
-s checkpoints/mt_bert \
--include-package utils \
--include-package multi_task \
--include-package modules \
--include-package metrics \
--file-friendly-logging \
config/multi_task_stack_buffer.json