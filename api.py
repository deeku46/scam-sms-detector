"""A tiny HTTP API around the trained model, ready to plug into a WhatsApp or Telegram bot.

Run:   uvicorn api:app --reload
Call:  curl -X POST localhost:8000/check -H "Content-Type: application/json" -d '{"text": "You won Rs 50,000!"}'
"""
import sys, joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent / "training"))
from normalise import normalise

bundle = joblib.load(Path(__file__).parent / "model.joblib")
vec, clf = bundle["vec"], bundle["clf"]
vocab = {i: t for t, i in vec.vocabulary_.items()}
app = FastAPI(title="Scam SMS Detector")

class Message(BaseModel):
    text: str

@app.post("/check")
def check(msg: Message):
    v = vec.transform([normalise(msg.text)])
    p = float(clf.predict_proba(v)[0, 1])
    contrib = sorted(((vocab[i], float(v[0, i] * clf.coef_[0][i])) for i in v.nonzero()[1]), key=lambda t: -t[1])
    return {"scam_probability": round(p, 4), "is_scam": p >= 0.5,
            "top_signals": [{"term": t, "weight": round(w, 3)} for t, w in contrib[:5] if w > 0]}
