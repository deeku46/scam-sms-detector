"""Train the scam SMS classifier and export it for the browser.

Data: SMS Spam Collection v1 (Almeida & Hidalgo, 2011, UCI ML repository): 5,572 real
SMS messages, 747 labelled spam. 80/20 stratified split; the test set is touched once.
Models compared with 5-fold CV on the training set: Naive Bayes, logistic regression, linear SVM.
Shipped model: TF-IDF (words + word pairs, plus placeholder tokens for links, phone numbers
and money) with logistic regression, because every word's weight can be shown to the user.
"""
import json, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from normalise import normalise

TOKEN = r"(?u)\b_?\w\w+_?\b"
df = pd.read_csv("../data/sms.tsv", sep="\t", header=None, names=["label", "text"])
df = df.drop_duplicates("text")
y = (df.label == "spam").astype(int).values
X = df.text.map(normalise).values
Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, df.text.values, test_size=0.2, stratify=y, random_state=42)

vec = TfidfVectorizer(token_pattern=TOKEN, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
Vtr = vec.fit_transform(Xtr)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
compare = {}
for name, m in [("Naive Bayes", MultinomialNB(alpha=0.1)), ("Logistic regression", LogisticRegression(C=20, max_iter=3000, class_weight="balanced")),
                ("Linear SVM", LinearSVC(C=1, class_weight="balanced"))]:
    compare[name] = round(float(cross_val_score(m, Vtr, ytr, cv=cv, scoring="f1").mean()), 4)
print("5-fold CV F1:", compare)

clf = LogisticRegression(C=20, max_iter=3000, class_weight="balanced").fit(Vtr, ytr)
Vte = vec.transform(Xte)
p = clf.predict_proba(Vte)[:, 1]
pred = (p >= 0.5).astype(int)
cm = confusion_matrix(yte, pred)
res = {"messages": int(len(df)), "spam": int(y.sum()), "train": int(len(ytr)), "test": int(len(yte)),
       "cv_f1": compare, "precision": round(float(precision_score(yte, pred)), 4),
       "recall": round(float(recall_score(yte, pred)), 4), "f1": round(float(f1_score(yte, pred)), 4),
       "auc": round(float(roc_auc_score(yte, p)), 4), "cm": cm.tolist(), "vocab": int(len(vec.vocabulary_))}
print(json.dumps(res))

# a few test messages for the page, mixed, never seen in training
rng = np.random.default_rng(3)
# keep the public page clean: skip adult-content spam and very short or slang-heavy messages
BLOCK = re.compile(r"sex|xxx|babe|bitch|nasty|horny|naughty|dating|married local|flirt|wet|hot |\bass\b", re.I)
ok = np.array([not BLOCK.search(t) and 40 <= len(t) <= 160 for t in tte])
idx = list(rng.choice(np.where((yte == 1) & ok)[0], 2, replace=False)) + list(rng.choice(np.where((yte == 0) & ok)[0], 2, replace=False))
examples = [{"text": str(tte[i]), "spam": int(yte[i])} for i in idx]
# mistakes, to be honest about failure modes
fn = [str(tte[i]) for i in np.where((yte == 1) & (pred == 0) & ok)[0][:3]]
fp = [str(tte[i]) for i in np.where((yte == 0) & (pred == 1) & ok)[0][:3]]
res["missed"] = fn; res["false_alarms"] = fp

terms = {t: [round(float(vec.idf_[i]), 4), round(float(clf.coef_[0][i]), 4)] for t, i in vec.vocabulary_.items()}
top_spam = sorted(terms, key=lambda t: -terms[t][1])[:15]
top_ham = [t for t in sorted(terms, key=lambda t: terms[t][1]) if not re.search(r"\b(lt|gt)\b", t)][:15]   # skip &lt;#&gt; artefacts
res["top_spam"] = top_spam; res["top_ham"] = top_ham
with open("../model.js", "w", encoding="utf-8") as f:
    f.write("window.SMS = " + json.dumps({"terms": terms, "bias": round(float(clf.intercept_[0]), 4),
            "metrics": res, "examples": examples}, ensure_ascii=False, separators=(",", ":")) + ";\n")

import joblib; joblib.dump({"vec": vec, "clf": clf}, "../model.joblib")
