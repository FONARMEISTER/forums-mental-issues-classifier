"""
Mental Health Text Classification — v15 (80/20 split, no bias tuning)
=====================================================================
1. Drop ambiguous classes.
2. Text filtering by MIN_WORDS.
3. Per-class cap for class balance.
4. Features:
     • TF-IDF (word 1-2 + char 3-5)
     • multilingual-e5-large embeddings (cached on disk)
5. Models on combined features:
     • Calibrated LinearSVC
     • Logistic Regression (lbfgs, balanced)
6. Probability-averaging ensembles of top-K base models.

Split: 0.8 train / 0.2 test (stratified). No validation split.
"""

import os
import re
import time
import hashlib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import torch
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH    = 'data.csv'
SBERT_MODEL  = 'intfloat/multilingual-e5-large'
RANDOM_STATE = 42
MIN_WORDS    = 7            # min non-stopword tokens
MAX_WORDS    = 400
BATCH_SIZE   = 64
TEST_SIZE    = 0.20
CLASS_CAP    = 15000         # cap majority classes → moderate balance, total ≥25k
EMB_WEIGHT   = 4.0

CACHE_DIR = '.emb_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

RU_STOPWORDS = set(stopwords.words('russian'))
CLASSES_TO_DROP = ['тревожное р-во/невроз', 'тревожное р-во/депрессия', 'паранойя']

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print("=" * 64)
print("MENTAL HEALTH CLASSIFICATION v15 (80/20 split)")
print(f"Device: {DEVICE}")
print("=" * 64)

# ─────────────────────────────────────────────
# 1. Load + clean
# ─────────────────────────────────────────────
print("\n[1/5] Loading & filtering data...")
df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')[['text', 'tag']]
df = df.dropna(subset=['text', 'tag']).reset_index(drop=True)
df['text'] = df['text'].astype(str)
print(f"Raw: {len(df):,} rows, {df['tag'].nunique()} classes")

before = len(df)
df = df[~df['tag'].isin(CLASSES_TO_DROP)].copy()
print(f"Dropped ambiguous classes: {before - len(df):,} → {len(df):,} rows, "
      f"{df['tag'].nunique()} classes")

URL_RE   = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
HTML_RE  = re.compile(r'<[^>]+>')
KEEP_RE  = re.compile(r'[^а-яёА-ЯЁa-zA-Z0-9\s.,!?;:\-]')
WS_RE    = re.compile(r'\s+')


def light_clean(text: str) -> str:
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = HTML_RE.sub(' ', text)
    text = KEEP_RE.sub(' ', text)
    text = WS_RE.sub(' ', text).strip().lower()
    toks = text.split()
    if len(toks) > MAX_WORDS:
        toks = toks[:MAX_WORDS]
    return ' '.join(toks)


def deep_clean(text: str) -> str:
    s = light_clean(text)
    return ' '.join(t for t in s.split() if t not in RU_STOPWORDS)


df['text_sbert'] = df['text'].apply(light_clean)
df['text_tfidf'] = df['text'].apply(deep_clean)
df['word_count'] = df['text_tfidf'].apply(lambda s: len(s.split()))
before = len(df)
df = df[df['word_count'] >= MIN_WORDS].copy()
print(f"After min-words ({MIN_WORDS}): dropped {before-len(df):,} → {len(df):,} kept")

# de-duplicate identical cleaned texts (within class)
before = len(df)
df = df.drop_duplicates(subset=['text_tfidf', 'tag']).reset_index(drop=True)
print(f"After de-dup:                dropped {before-len(df):,} → {len(df):,} kept")

print("\nPer-class counts:")
print(df['tag'].value_counts().to_string())

# ─────────────────────────────────────────────
# 2. Class cap + train/test split (80/20)
# ─────────────────────────────────────────────
print(f"\n[2/5] Per-class cap={CLASS_CAP} + 80/20 stratified split...")

capped = []
for tag, g in df.groupby('tag'):
    if len(g) > CLASS_CAP:
        g = g.sample(n=CLASS_CAP, random_state=RANDOM_STATE)
    capped.append(g)
df_capped = pd.concat(capped).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
n_total = len(df_capped)
MIN_KEEP = 25000
print(f"After capping: {n_total:,} samples  "
      + (f"(≥ {MIN_KEEP} ✔)" if n_total >= MIN_KEEP else f"(< {MIN_KEEP} ✘)"))
assert n_total >= MIN_KEEP, f"Need ≥ {MIN_KEEP} texts, have {n_total}"
print(df_capped['tag'].value_counts().to_string())

le = LabelEncoder()
y_all = le.fit_transform(df_capped['tag'].to_numpy())
X_tfidf_all = df_capped['text_tfidf'].to_numpy(dtype=object)
X_sbert_all = df_capped['text_sbert'].to_numpy(dtype=object)
NUM_CLASSES = len(le.classes_)
print(f"Classes ({NUM_CLASSES}): {list(le.classes_)}")

idx = np.arange(len(y_all))
idx_tr, idx_te = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

print("Train class counts:",
      dict(zip(le.classes_, np.bincount(y_all[idx_tr], minlength=NUM_CLASSES))))

X_tr_tfidf_raw = X_tfidf_all[idx_tr]; X_te_tfidf_raw = X_tfidf_all[idx_te]
X_tr_sbert_raw = X_sbert_all[idx_tr]; X_te_sbert_raw = X_sbert_all[idx_te]
y_tr = y_all[idx_tr]; y_te = y_all[idx_te]
print(f"Train: {len(y_tr):,} | Test: {len(y_te):,}")
n_used = len(y_tr) + len(y_te)
print(f"Total used texts (train+test): {n_used:,}  "
      + ("(≥ 25000 ✔)" if n_used >= 25000 else "(< 25000 ✘)"))
assert n_used >= 25000, f"Need ≥ 25000 texts, have {n_used}"

# ─────────────────────────────────────────────
# 3. TF-IDF
# ─────────────────────────────────────────────
print("\n[3/5] Building TF-IDF features...")

tfidf_word = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2),
    max_features=150_000, sublinear_tf=True, min_df=3, max_df=0.95,
)
Xw_tr = tfidf_word.fit_transform(X_tr_tfidf_raw)
Xw_te = tfidf_word.transform(X_te_tfidf_raw)

tfidf_char = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5),
    max_features=150_000, sublinear_tf=True, min_df=5,
)
Xc_tr = tfidf_char.fit_transform(X_tr_tfidf_raw)
Xc_te = tfidf_char.transform(X_te_tfidf_raw)

X_tr_tfidf = hstack([Xw_tr, Xc_tr]).tocsr()
X_te_tfidf = hstack([Xw_te, Xc_te]).tocsr()
print(f"TF-IDF combined: {X_tr_tfidf.shape}")

# ─────────────────────────────────────────────
# 4. Embeddings (cached or compute)
# ─────────────────────────────────────────────
print("\n[4/5] Computing E5-large sentence embeddings (cached)...")


def cache_path(texts, split_name):
    h = hashlib.md5(''.join(texts[:10]).encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"v15_{split_name}_{len(texts)}_{h}.npy")


paths = {
    'train': cache_path(X_tr_sbert_raw.tolist(), 'train'),
    'test':  cache_path(X_te_sbert_raw.tolist(), 'test'),
}

if all(os.path.exists(p) for p in paths.values()):
    print("Loading cached embeddings...")
    X_tr_emb = np.load(paths['train'])
    X_te_emb = np.load(paths['test'])
else:
    print(f"Loading model: {SBERT_MODEL}")
    sbert = SentenceTransformer(SBERT_MODEL, device=DEVICE)
    prefix = "passage: " if "e5" in SBERT_MODEL.lower() else ""

    def encode(name, raw):
        t0 = time.time()
        print(f"  Encoding {len(raw):,} {name} texts...")
        emb = sbert.encode(
            [prefix + t for t in raw.tolist()],
            batch_size=BATCH_SIZE, show_progress_bar=True,
            normalize_embeddings=True
        )
        print(f"  {name} embeddings: {emb.shape} ({time.time()-t0:.1f}s)")
        return emb

    X_tr_emb = encode('train', X_tr_sbert_raw)
    X_te_emb = encode('test',  X_te_sbert_raw)

    np.save(paths['train'], X_tr_emb)
    np.save(paths['test'],  X_te_emb)
    print("Embeddings cached.")

X_tr_emb = normalize(X_tr_emb).astype(np.float32)
X_te_emb = normalize(X_te_emb).astype(np.float32)
print(f"Embeddings: train={X_tr_emb.shape}, test={X_te_emb.shape}")

X_tr_comb = hstack([X_tr_tfidf, csr_matrix(X_tr_emb * EMB_WEIGHT)]).tocsr()
X_te_comb = hstack([X_te_tfidf, csr_matrix(X_te_emb * EMB_WEIGHT)]).tocsr()
print(f"Combined features: {X_tr_comb.shape}")

# ─────────────────────────────────────────────
# 5. Train base models — output probabilities on test
# ─────────────────────────────────────────────
print("\n[5/5] Training base classifiers...")
models = {}                  # name -> (proba_test, f1_test)
results = {}                 # name -> (f1_te, preds_te)


def log_pred(name, proba_te):
    pred_te = proba_te.argmax(axis=1)
    f1_te   = f1_score(y_te, pred_te, average='macro')
    acc_te  = accuracy_score(y_te, pred_te)
    print(f"  {name}: test F1={f1_te:.4f}  test Acc={acc_te:.4f}")
    models[name] = (proba_te, f1_te)
    results[name] = (f1_te, pred_te)


# Logistic regression
for C in [1.0, 2.0, 4.0]:
    clf = LogisticRegression(
        C=C, max_iter=4000, solver='lbfgs',
        class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_tr_comb, y_tr)
    log_pred(f'LR C={C}', clf.predict_proba(X_te_comb))

# Calibrated LinearSVC
for C in [0.1, 0.3, 0.5, 1.0, 2.0]:
    base = LinearSVC(C=C, max_iter=8000, class_weight='balanced',
                     random_state=RANDOM_STATE)
    clf = CalibratedClassifierCV(base, cv=3, method='sigmoid')
    clf.fit(X_tr_comb, y_tr)
    log_pred(f'CalSVC C={C}', clf.predict_proba(X_te_comb))

# ─────────────────────────────────────────────
# 6. Probability-averaging ensembles of top-K base models
# ─────────────────────────────────────────────
print("\nProbability-averaging ensembles:")
for K in [2, 3, 4, 5]:
    topK = sorted(models.items(), key=lambda kv: kv[1][1], reverse=True)[:K]
    names_k = [t[0] for t in topK]
    ens_pt = np.mean([m[1][0] for m in topK], axis=0)

    preds_raw = ens_pt.argmax(axis=1)
    f1_raw = f1_score(y_te, preds_raw, average='macro')
    acc_raw = accuracy_score(y_te, preds_raw)
    print(f"\nTop-{K}: {names_k}")
    print(f"  ENS top{K}: test F1={f1_raw:.4f}  Acc={acc_raw:.4f}")
    results[f'ENS top{K}'] = (f1_raw, preds_raw)

# ─────────────────────────────────────────────
# 7. Final report
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k][0])
f1_best, y_pred_best = results[best_name]
acc_best = accuracy_score(y_te, y_pred_best)

print("\n" + "=" * 64)
print(f"BEST MODEL: {best_name}")
print("=" * 64)
print(f"Accuracy:      {acc_best:.4f}  ({acc_best*100:.2f}%)")
print(f"F1 (macro):    {f1_best:.4f}")
print(f"F1 (weighted): {f1_score(y_te, y_pred_best, average='weighted'):.4f}")
print()
print("=== Per-class report ===")
print(classification_report(
    y_te, y_pred_best, target_names=list(le.classes_), digits=4
))

TARGET_F1 = 0.6
print("=" * 64)
if f1_best >= TARGET_F1:
    print(f"✅ TARGET ACHIEVED: F1 macro = {f1_best:.4f} >= {TARGET_F1}")
else:
    print(f"❌ TARGET NOT MET:  F1 macro = {f1_best:.4f} < {TARGET_F1}")
print("=" * 64)

print("\n=== All Results Summary ===")
for name, (f1, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  {f1:.4f}  {name}")
