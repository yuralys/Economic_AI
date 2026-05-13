import torch
import torch.nn as nn
import pickle
from transformers import AutoModel, AutoTokenizer

# ======================
# CONFIG
# ======================
MODEL_PATH = "saved_model"
MAX_LEN = 512
THRESHOLD = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 1. Загружаем токенизатор (из локальной папки)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, fix_mistral_regex=True)

# 2. Загружаем mlb (список тикеров)
with open(f"{MODEL_PATH}/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# 3. Определяем класс модели (должен совпадать с обучением)
class NewsModel(nn.Module):
    def __init__(self, encoder_path, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
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

# 4. Создаём модель (энкодер загружается из MODEL_PATH)
model = NewsModel(MODEL_PATH, len(mlb.classes_))

# 5. Загружаем веса голов (без энкодера)
state_dict = torch.load(f"{MODEL_PATH}/model.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()

print("Модель загружена")

# 6. Функция предсказания
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

# 7. Цикл ввода новостей
if __name__ == "__main__":
    while True:
        text = input("\nВведите новость: ")
        if text.lower() in ["exit", "quit"]:
            break
        score, tickers = predict(text)
        print(f"Score: {score}")
        print(f"Tickers: {tickers}")