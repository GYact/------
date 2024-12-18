import pandas as pd
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import torch

# Plutchikの8つの基本感情
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']

@st.cache_resource
def load_model_and_tokenizer():
    """モデルとトークナイザをロード"""
    checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=8)
    return model, tokenizer

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def analyze_emotion(text, model, tokenizer):
    """感情分析"""
    model.eval()
    tokens = tokenizer(text, truncation=True, return_tensors="pt")
    with torch.no_grad():
        preds = model(**tokens)
    prob = np_softmax(preds.logits.cpu().numpy()[0])
    return {n: p for n, p in zip(emotion_names, prob)}

# Streamlit UI
st.title("感情分析アプリ")
st.write("文章を入力すると、Plutchikの8つの基本感情に基づいた感情分析を行います。")

# モデルとトークナイザのロード
model, tokenizer = load_model_and_tokenizer()

# ユーザー入力
text_input = st.text_area("文章を入力してください:")

if st.button("感情を分析する"):
    if text_input.strip():
        result = analyze_emotion(text_input, model, tokenizer)
        st.write("分析結果:")
        st.bar_chart(result)
    else:
        st.warning("文章を入力してください。")
