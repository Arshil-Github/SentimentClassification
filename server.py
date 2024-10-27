import joblib
import numpy as np
import fastapi
import re
import tensorflow as tf
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = fastapi.FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins (use cautiously)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

m = joblib.load("gru_sentiment_model.pkl")
tokenizer = joblib.load("tokenizer.pkl")

def preprocess_text(text, tokenizer=tokenizer):
    def clean_string(text):
        cleaned_text = re.sub(r"[^a-zA-Z0-9., ]", "", text)
        return cleaned_text
    tokens = tokenizer.texts_to_sequences([clean_string(text)])
    padded = tf.keras.preprocessing.sequence.pad_sequences(tokens, padding='post', maxlen=400)
    return padded

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text)
    prediction = m.predict(preprocessed_text)
    return "Positive" if prediction > 0.5 else "Negative"


@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Prediction API"}

@app.post("/predict")
def predict(parameters : dict):
    text = parameters["text"]
    print(text)
    prediction = predict_sentiment(text)
    return {"prediction": str(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, port=8000)