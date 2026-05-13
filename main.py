# main.py
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pickle
import os

# ======================
# CONFIG
# ======================
# Локальный путь к скачанной модели
LOCAL_MODEL_PATH = "./my_rubert_model"
MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

print("CUDA available:", torch.cuda.is_available())  # Должно быть True
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))

# ======================
# ЗАГРУЗКА ДАННЫХ
# ======================
df = pd.read_csv("data.tsv", sep="\t")

for col in ["title", "summary"]:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)
    else:
        df[col] = ""

df["text"] = (df["title"] + " " + df["summary"]).str.strip()
df = df[df["text"].str.len() > 0].reset_index(drop=True)

def parse_tickers(x):
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            return val if isinstance(val, list) else []
        except:
            return []
    return []

df["tickers"] = df["tickers"].apply(parse_tickers)
df["score"] = pd.to_numeric(df["score"], errors="coerce")
df = df[df["score"].notna()].reset_index(drop=True)

mlb = MultiLabelBinarizer()
y_tickers = mlb.fit_transform(df["tickers"])
y_score = df["score"].values.astype("float32")

X_train, X_val, y_score_train, y_score_val, y_tick_train, y_tick_val = train_test_split(
    df["text"].tolist(),
    y_score,
    y_tickers,
    test_size=0.1,
    random_state=42
)

# ======================
# ТОКЕНИЗАТОР (из локальной папки)
# ======================
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

# ======================
# DATASET
# ======================
class NewsDataset(Dataset):
    def __init__(self, texts, scores, tickers):
        self.texts = texts
        self.scores = scores
        self.tickers = tickers

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score": torch.tensor(self.scores[idx], dtype=torch.float32),
            "tickers": torch.tensor(self.tickers[idx], dtype=torch.float32),
        }

train_dataset = NewsDataset(X_train, y_score_train, y_tick_train)
val_dataset = NewsDataset(X_val, y_score_val, y_tick_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ======================
# МОДЕЛЬ (энкодер из локальной папки)
# ======================
class NewsModel(nn.Module):
    def __init__(self, encoder_path, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)   # без use_safetensors
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.regressor = nn.Linear(hidden_size, 1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        score = torch.tanh(self.regressor(cls)).squeeze(-1)
        tickers = torch.sigmoid(self.classifier(cls))
        return score, tickers

model = NewsModel(LOCAL_MODEL_PATH, y_tickers.shape[1]).to(DEVICE)

# ======================
# ОБУЧЕНИЕ
# ======================
loss_reg = nn.MSELoss()
loss_cls = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        scores = batch["score"].to(DEVICE)
        tickers = batch["tickers"].to(DEVICE)

        pred_score, pred_tickers = model(input_ids, attention_mask)
        loss1 = loss_reg(pred_score, scores)
        loss2 = loss_cls(pred_tickers, tickers)
        loss = loss1 + loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} train loss: {total_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            scores = batch["score"].to(DEVICE)
            tickers = batch["tickers"].to(DEVICE)
            pred_score, pred_tickers = model(input_ids, attention_mask)
            loss1 = loss_reg(pred_score, scores)
            loss2 = loss_cls(pred_tickers, tickers)
            val_loss += (loss1 + loss2).item()
    print(f"Epoch {epoch} val loss: {val_loss:.4f}")

# ======================
# ТЕСТОВЫЙ ПРИМЕР
# ======================
def predict(text):
    model.eval()
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

test_text = "Компания показала рост прибыли и объявила дивиденды"
score, tickers = predict(test_text)
print("\nPrediction on test text:")
print("Score:", score)
print("Tickers:", tickers)

# ======================
# СОХРАНЕНИЕ ВСЕЙ МОДЕЛИ В saved_model
# ======================
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 1. Сохраняем веса голов (regressor, classifier, dropout)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pt"))

# 2. Сохраняем энкодер (базовую модель) — копирует все файлы из my_rubert_model
model.encoder.save_pretrained(SAVE_DIR)

# 3. Сохраняем токенизатор
tokenizer.save_pretrained(SAVE_DIR)

# 4. Сохраняем mlb
with open(os.path.join(SAVE_DIR, "mlb.pkl"), "wb") as f:
    pickle.dump(mlb, f)

print(f"\nМодель полностью сохранена в {SAVE_DIR}")
print("Содержимое папки:", os.listdir(SAVE_DIR))