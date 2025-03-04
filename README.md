---
title: Dog Breed Recognition
emoji: 🌍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Dog Breed Recognition with LLaVA

This app uses the LLaVA-v1.5-7B model fine-tuned on the Stanford Dogs dataset to identify dog breeds from images.

## How to use
1. Click the "Load Model" button (this may take 1-2 minutes to load)
2. Upload a dog image
3. Optionally, customize your question
4. Click "Identify Breed"

## About the model
This app uses [YuchengShi/LLaVA-v1.5-7B-Stanford-Dogs](https://huggingface.co/YuchengShi/LLaVA-v1.5-7B-Stanford-Dogs), a vision-language model fine-tuned specifically for dog breed identification.
