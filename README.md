# 🤗 empathetic-multilabel-emotion-classification

다중 감정 인식(Multi-label Emotion Classification)을 위해 BERT 기반 모델을 활용하여 EmpatheticDialogues 데이터셋에서 감정 예측을 수행하는 프로젝트입니다.  
NRC 감정 사전을 기반으로 단일 감정을 다중 감정으로 확장하여 보다 정밀한 감정 분석이 가능하도록 설계되었습니다.

---

## 💡 프로젝트 개요

- **데이터셋**: [EmpatheticDialogues](https://huggingface.co/datasets/empathetic_dialogues)
- **모델**: `bert-base-uncased` fine-tuned with multi-label setup
- **레이블링 방식**: NRC Emotion Lexicon 기반 감정 확장
- **분류 방식**: Binary Cross Entropy + Sigmoid → 다중 감정 예측
- **감정 수**: 총 8~28개 감정 레이블 (확장 가능)

---

## 📁 프로젝트 구조

```
empathetic-multilabel-emotion-classification/
├── data/
│   └── empatheticdialogues_with_multitags.csv
├── nrc_emotion_dict.json
├── multilabel_emotion_bert.ipynb
├── multilabel_emotion_bert.py
├── README.md
└── requirements.txt
```

---

## ⚙️ 주요 기술 스택

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- NLTK (for optional preprocessing)
- seaborn, matplotlib

---

## 🚀 실행 방법

1. 모델 훈련:

```bash
python multilabel_emotion_bert.py
```

2. 학습된 모델 불러오기 및 예측:

```python
model.load_state_dict(torch.load("multilabel_emotion_bert.pt"))
```

3. 시각화 결과 확인:
- 감정별 F1 score heatmap
- 감정 간 동시출현 분석 (co-occurrence matrix)

---

## 📊 모델 성능 요약

| 지표 | 결과 |
|------|------|
| Micro F1 Score | **0.988** |
| Precision / Recall (평균) | 0.99 / 0.99 |
| 주요 감정 정확도 | joy, trust, anticipation 등 0.99+ |

---

## 📈 시각화 결과 예시

### 감정별 Precision / Recall / F1 Score

![emotion-f1-heatmap](./assets/emotion_f1_heatmap.png)

### 감정 예측 분포

![emotion-distribution](./assets/emotion_bar_chart.png)

---

## 📌 포트폴리오 활용 포인트

- 다중 감정 처리 구조와 감정 사전 활용 방식
- BERT 기반 문맥 감정 추론
- 자연어에서 은유적·복합 감정 추론 능력 강화

---

## 📚 참고 자료

- [EmpatheticDialogues Dataset](https://huggingface.co/datasets/empathetic_dialogues)
- [NRC Emotion Lexicon (Mohammad & Turney, 2013)](http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

---

## 🙋‍♂️ 만든 사람

김준식 (Junsik Kim)  
AI/NLP 엔지니어, 감정 인식·추천 시스템 연구자
