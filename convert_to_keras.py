import os
import pickle
import tempfile
import shutil

import torch
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoConfig
from transformers import AutoTokenizer as PyTorchTokenizer

# ======================
# 1. Define the original PyTorch model architecture (same as in training)
# ======================
import torch.nn as nn
from transformers import AutoModel as PyTorchAutoModel

class NewsModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = PyTorchAutoModel.from_pretrained(model_name, use_safetensors=True)
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

# ======================
# 2. Load the saved PyTorch model and extract weights
# ======================
SAVE_DIR = "saved_model"            # directory containing model.pt, mlb.pkl, tokenizer/
DEVICE = torch.device("cpu")        # load on CPU

# Load mlb to know num_labels
with open(os.path.join(SAVE_DIR, "mlb.pkl"), "rb") as f:
    mlb = pickle.load(f)
num_labels = len(mlb.classes_)

# Load model architecture and state_dict
model_name = "DeepPavlov/rubert-base-cased"
pytorch_model = NewsModel(model_name, num_labels)
state_dict = torch.load(os.path.join(SAVE_DIR, "model.pt"), map_location="cpu")
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()

# Extract transformer (encoder) weights into a temporary directory
temp_dir = tempfile.mkdtemp()
# Save config.json from the original pretrained model
config = AutoConfig.from_pretrained(model_name)
config.save_pretrained(temp_dir)
# Save only the encoder's state_dict as pytorch_model.bin
encoder_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")}
torch.save(encoder_state_dict, os.path.join(temp_dir, "pytorch_model.bin"))

# ======================
# 3. Load the fine-tuned transformer weights into TensorFlow/Keras
# ======================
tf_transformer = TFAutoModel.from_pretrained(temp_dir, from_pt=True)  # loads fine-tuned weights
shutil.rmtree(temp_dir)  # clean up

# ======================
# 4. Build the complete Keras model
# ======================
class KerasNewsModel(tf.keras.Model):
    def __init__(self, transformer, num_labels):
        super().__init__()
        self.transformer = transformer
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.regressor = tf.keras.layers.Dense(1, activation='tanh', name='regressor')
        self.classifier = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='classifier')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]   # [CLS] token
        cls_token = self.dropout(cls_token)
        score = self.regressor(cls_token)
        score = tf.squeeze(score, axis=-1)               # shape (batch,)
        tickers = self.classifier(cls_token)
        return score, tickers

# Instantiate Keras model
keras_model = KerasNewsModel(tf_transformer, num_labels)

# ======================
# 5. Transfer the weights of the two heads (regressor & classifier)
# ======================
# PyTorch weights for regressor: weight shape (1, hidden_size), bias (1)
pytorch_reg_weight = state_dict['regressor.weight'].detach().numpy()   # (1, hidden)
pytorch_reg_bias = state_dict['regressor.bias'].detach().numpy()       # (1,)

# PyTorch classifier: weight shape (num_labels, hidden), bias (num_labels)
pytorch_cls_weight = state_dict['classifier.weight'].detach().numpy()  # (num_labels, hidden)
pytorch_cls_bias = state_dict['classifier.bias'].detach().numpy()      # (num_labels,)

# Assign to Keras layers
keras_model.regressor.set_weights([pytorch_reg_weight.T, pytorch_reg_bias])   # Keras expects (hidden, 1)
keras_model.classifier.set_weights([pytorch_cls_weight.T, pytorch_cls_bias])   # Keras expects (hidden, num_labels)

# ======================
# 6. Test the Keras model with a sample input (optional)
# ======================
# Load the same tokenizer (from saved_model/tokenizer) – but it's PyTorch tokenizer.
# For Keras, we can use the same tokenizer because it's just tokenization.
tokenizer = PyTorchTokenizer.from_pretrained(SAVE_DIR)   # loads the saved tokenizer

sample_text = "Компания показала рост прибыли и объявила дивиденды"
encoded = tokenizer(sample_text, return_tensors='tf', truncation=True, padding='max_length', max_length=256)
input_ids = encoded['input_ids']
attention_mask = encoded['attention_mask']

score, tickers = keras_model((input_ids, attention_mask))
print("Keras model test output:")
print("Score:", score.numpy())
print("Tickers (probabilities):", tickers.numpy())

# ======================
# 7. Save the Keras model in SavedModel format
# ======================
keras_model.save("keras_model", save_format='tf')
print("\nKeras model saved to 'keras_model' directory")

# Also save the MultiLabelBinarizer and tokenizer for later use
import pickle
with open("keras_model/mlb.pkl", "wb") as f:
    pickle.dump(mlb, f)
tokenizer.save_pretrained("keras_model")
print("Tokenizer and MultiLabelBinarizer saved inside 'keras_model'")