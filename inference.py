import torch
import torch.nn as nn
import numpy as np
import pickle
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = "saved_model"          #configs
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LEN = 256
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)



tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)       #load tokens



with open(f"{MODEL_PATH}/mlb.pkl", "rb") as f:      #load mlb
    mlb = pickle.load(f)



class NewsModel(nn.Module):                     #model class(like in main.py)
    def __init__(self, model_name, num_labels):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            use_safetensors=True
        )

        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)

        score = torch.tanh(self.regressor(cls)).squeeze(-1)
        tickers = torch.sigmoid(self.classifier(cls))

        return score, tickers



model = NewsModel(MODEL_NAME, len(mlb.classes_))            #load model
model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded")


def predict(text):
    encoding = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        score, tickers = model(input_ids, attention_mask)

    score = score.item()

    tickers = (tickers.cpu().numpy() > THRESHOLD).astype(int)
    ticker_names = mlb.inverse_transform(tickers)[0]

    return score, ticker_names




while True:
    text = input("\nВведите новость: ")

    if text.lower() in ["exit", "quit"]:
        break

    score, tickers = predict(text)

    print("\nРезультат:")
    print("Score:", round(score, 4))
    print("Tickers:", tickers)