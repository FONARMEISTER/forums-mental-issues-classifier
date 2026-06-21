"""
Mental Health Text Classification — v17
=======================================
1. Drop ambiguous classes.
2. Text filtering by MIN_WORDS.
3. Per-class cap for class balance.
4. Split base data into train/test. Augmented data is appended to train only.
5. Fine-tune the transformer classifier first.
6. Extract embeddings from the fine-tuned transformer encoder.
7. Train/evaluate SVM and MLP on fine-tuned transformer embeddings only.

Split: 0.8 train / 0.2 test (stratified). Augmented data is never used for test evaluation.
"""

import argparse
import gc
import inspect
import os
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import torch 
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize, LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_PATH = 'data_filtered.csv'
AUGMENTED_DATA_PATH = 'augmented_data.csv'
USE_AUGMENTATION = True
SBERT_MODEL = 'ai-forever/ru-en-RoSBERTa'
RANDOM_STATE = 42
MIN_WORDS = 5
MAX_WORDS = 1500
EMB_BATCH_SIZE = 64
TEST_SIZE = 0.20
CLASS_CAP = 12000

RUN_BERT_FINETUNE = True
BERT_FT_EPOCHS = 3
BERT_FT_BATCH = 64
BERT_FT_EVAL_BATCH = 64
BERT_FT_LR = 2e-5 * (BERT_FT_BATCH / 16) ** 0.5
BERT_FT_MAX_LEN = 256
BERT_FT_GRADIENT_CHECKPOINTING = False
BERT_FT_OUTPUT_DIR = '.bert_ft_out'

RU_STOPWORDS = set(stopwords.words('russian'))

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune transformer, extract embeddings, then train SVM/MLP classifiers.'
    )
    parser.add_argument(
        '--epochs', '--bert-ft-epochs',
        dest='bert_ft_epochs',
        type=int,
        default=BERT_FT_EPOCHS,
        help='Number of transformer fine-tuning epochs.',
    )
    parser.add_argument(
        '--use-augmentation',
        dest='use_augmentation',
        type=int,
        choices=(0, 1),
        default=int(USE_AUGMENTATION),
        help='Whether to append augmented data to the training split only: 1=yes, 0=no.',
    )
    parser.add_argument(
        '--sbert-model', '--model',
        dest='sbert_model',
        type=str,
        default=SBERT_MODEL,
        help='HuggingFace model name or local path used as the transformer encoder/classifier.',
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[WARN] Ignoring unknown CLI arguments: {unknown}")
    if args.bert_ft_epochs < 1:
        raise ValueError('--epochs must be >= 1')
    return args


CLI_ARGS = parse_args()
BERT_FT_EPOCHS = CLI_ARGS.bert_ft_epochs
USE_AUGMENTATION = bool(CLI_ARGS.use_augmentation)
SBERT_MODEL = CLI_ARGS.sbert_model
BERT_FT_OUTPUT_DIR = os.path.join(
    BERT_FT_OUTPUT_DIR,
    f"epochs_{BERT_FT_EPOCHS}_aug_{int(USE_AUGMENTATION)}",
)

CLASSES_TO_DROP = ['тревожное р-во/депрессия', 'паранойя', 'тревожное р-во/невроз']

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

print("=" * 64)
print("MENTAL HEALTH CLASSIFICATION v17 (fine-tune → embeddings → SVM/MLP)")
print(f"Device:          {DEVICE}")
print(f"Transformer:     {SBERT_MODEL}")
print(f"FT epochs:       {BERT_FT_EPOCHS}")
print(f"Use augment:     {USE_AUGMENTATION}")
print(f"Augment path:    {AUGMENTED_DATA_PATH}")
print(f"FT output dir:   {BERT_FT_OUTPUT_DIR}")
print("=" * 64)

# ─────────────────────────────────────────────
# 1. Load + clean
# ─────────────────────────────────────────────
print("\n[1/6] Loading & filtering data...")
df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')[['text', 'tag']]
df = df.dropna(subset=['text', 'tag']).reset_index(drop=True)
df['text'] = df['text'].astype(str)
print(f"Raw: {len(df):,} rows, {df['tag'].nunique()} classes")

before = len(df)
df = df[~df['tag'].isin(CLASSES_TO_DROP)].copy()
print(f"Dropped ambiguous classes: {before - len(df):,} → {len(df):,} rows, "
      f"{df['tag'].nunique()} classes")

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
HTML_RE = re.compile(r'<[^>]+>')
KEEP_RE = re.compile(r'[^а-яёА-ЯЁa-zA-Z0-9\s.,!?;:\-]')
WS_RE = re.compile(r'\s+')


def text_clean(text: str) -> str:
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = HTML_RE.sub(' ', text)
    text = KEEP_RE.sub(' ', text)
    text = WS_RE.sub(' ', text).strip().lower()
    toks = text.split()
    if len(toks) > MAX_WORDS:
        toks = toks[:MAX_WORDS]
    return ' '.join(t for t in toks if t not in RU_STOPWORDS)


def clean_dataframe(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    frame = frame.dropna(subset=['text', 'tag']).reset_index(drop=True)
    frame['text'] = frame['text'].astype(str)
    frame['text_sbert'] = frame['text'].apply(text_clean)
    frame['word_count'] = frame['text_sbert'].apply(lambda s: len(s.split()))

    before_local = len(frame)
    frame = frame[frame['word_count'] >= MIN_WORDS].copy()
    print(f"{name} after min-words ({MIN_WORDS}): dropped {before_local-len(frame):,} → {len(frame):,} kept")

    before_local = len(frame)
    frame = frame.drop_duplicates(subset=['text_sbert', 'tag']).reset_index(drop=True)
    print(f"{name} after de-dup:                dropped {before_local-len(frame):,} → {len(frame):,} kept")
    return frame


df = clean_dataframe(df, 'Base')

print("\nPer-class counts:")
print(df['tag'].value_counts().to_string())

# ─────────────────────────────────────────────
# 2. Class cap + train/test split + train-only augmentation
# ─────────────────────────────────────────────
print(f"\n[2/6] Per-class cap={CLASS_CAP} + 80/20 stratified split + optional train-only augmentation...")

capped = []
for tag, g in df.groupby('tag'):
    if len(g) > CLASS_CAP:
        g = g.sample(n=CLASS_CAP, random_state=RANDOM_STATE)
    capped.append(g)
df_capped = pd.concat(capped).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
n_total = len(df_capped)
MIN_KEEP = 25000
print(f"After capping: {n_total:,} samples")
assert n_total >= MIN_KEEP, f"Need ≥ {MIN_KEEP} texts, have {n_total}"
print(df_capped['tag'].value_counts().to_string())

le = LabelEncoder()
y_all = le.fit_transform(df_capped['tag'].to_numpy())
X_sbert_all = df_capped['text_sbert'].to_numpy(dtype=object)
NUM_CLASSES = len(le.classes_)
print(f"Classes ({NUM_CLASSES}): {list(le.classes_)}")

idx = np.arange(len(y_all))
idx_tr, idx_te = train_test_split(
    idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_all
)

X_tr_sbert_raw = X_sbert_all[idx_tr]
X_te_sbert_raw = X_sbert_all[idx_te]
y_tr = y_all[idx_tr]
y_te = y_all[idx_te]
base_train_size = len(y_tr)
base_test_size = len(y_te)

print("Base train class counts:",
      dict(zip(le.classes_, np.bincount(y_tr, minlength=NUM_CLASSES))))

if USE_AUGMENTATION and AUGMENTED_DATA_PATH and os.path.exists(AUGMENTED_DATA_PATH):
    print(f"\nLoading train-only augmented data: {AUGMENTED_DATA_PATH}")
    df_aug = pd.read_csv(AUGMENTED_DATA_PATH, sep=';', encoding='utf-8')[['text', 'tag']]
    print(f"Augmented raw: {len(df_aug):,} rows, {df_aug['tag'].nunique()} classes")

    before = len(df_aug)
    df_aug = df_aug[~df_aug['tag'].isin(CLASSES_TO_DROP)].copy()
    print(f"Augmented dropped ambiguous classes: {before - len(df_aug):,} → {len(df_aug):,} rows")

    df_aug = clean_dataframe(df_aug, 'Augmented')

    unknown_aug_tags = sorted(set(df_aug['tag']) - set(le.classes_))
    if unknown_aug_tags:
        before = len(df_aug)
        df_aug = df_aug[df_aug['tag'].isin(le.classes_)].copy()
        print(f"Augmented dropped unknown classes {unknown_aug_tags}: {before-len(df_aug):,} rows")

    if len(df_aug) > 0:
        y_aug = le.transform(df_aug['tag'].to_numpy())
        X_tr_sbert_raw = np.concatenate([
            X_tr_sbert_raw,
            df_aug['text_sbert'].to_numpy(dtype=object),
        ])
        y_tr = np.concatenate([y_tr, y_aug])
        print("Augmented train class counts:",
              dict(zip(le.classes_, np.bincount(y_aug, minlength=NUM_CLASSES))))
        print(f"Merged {len(y_aug):,} augmented rows into TRAIN only.")
    else:
        print("No augmented rows remained after filtering; training uses base data only.")
elif USE_AUGMENTATION and AUGMENTED_DATA_PATH:
    print(f"\n[WARN] Augmented data file not found: {AUGMENTED_DATA_PATH}; training uses base data only.")
else:
    print("\nAugmentation disabled by CLI/config; training uses base data only.")

print(f"Train: {len(y_tr):,} ({base_train_size:,} base + {len(y_tr)-base_train_size:,} augmented) | "
      f"Test: {len(y_te):,} (base only)")
print("Final train class counts:",
      dict(zip(le.classes_, np.bincount(y_tr, minlength=NUM_CLASSES))))
print("Test class counts (evaluation set, no augmented data):",
      dict(zip(le.classes_, np.bincount(y_te, minlength=NUM_CLASSES))))
n_used = base_train_size + base_test_size
print(f"Total base texts used for split/evaluation accounting: {n_used:,}")
assert n_used >= MIN_KEEP, f"Need ≥ MIN_KEEP texts, have {n_used}"
assert y_tr.min() >= 0 and y_tr.max() < NUM_CLASSES
assert y_te.min() >= 0 and y_te.max() < NUM_CLASSES

# ─────────────────────────────────────────────
# Shared transformer dataset / trainer helpers
# ─────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            padding=False,
            return_tensors=None,
        )
        if self.labels is not None:
            enc['labels'] = int(self.labels[i])
        return enc


def clear_accelerator_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_training_args(ft_device: str):
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    args_kwargs = dict(
        output_dir=BERT_FT_OUTPUT_DIR,
        num_train_epochs=BERT_FT_EPOCHS,
        per_device_train_batch_size=BERT_FT_BATCH,
        per_device_eval_batch_size=BERT_FT_EVAL_BATCH,
        learning_rate=BERT_FT_LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        save_strategy='no',
        report_to='none',
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        gradient_checkpointing=BERT_FT_GRADIENT_CHECKPOINTING,
    )
    if ft_device == 'mps':
        if 'use_mps_device' in training_args_params:
            args_kwargs['use_mps_device'] = True
    elif ft_device == 'cpu':
        if 'use_cpu' in training_args_params:
            args_kwargs['use_cpu'] = True
        elif 'no_cuda' in training_args_params:
            args_kwargs['no_cuda'] = True
        if 'use_mps_device' in training_args_params:
            args_kwargs['use_mps_device'] = False
    return TrainingArguments(**args_kwargs)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
    }


# ─────────────────────────────────────────────
# 3. Fine-tune BERT first
# ─────────────────────────────────────────────
print("\n[3/6] Fine-tuning transformer classifier before embedding extraction...")
tokenizer = AutoTokenizer.from_pretrained(SBERT_MODEL)
ft_device = DEVICE if DEVICE in ('cuda', 'mps') else 'cpu'
print(f"Fine-tune requested device: {ft_device}")

bert_clf = AutoModelForSequenceClassification.from_pretrained(
    SBERT_MODEL,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True,
)
if BERT_FT_GRADIENT_CHECKPOINTING:
    bert_clf.gradient_checkpointing_enable()
    if hasattr(bert_clf.config, 'use_cache'):
        bert_clf.config.use_cache = False
    print("Gradient checkpointing enabled.")
else:
    print("Gradient checkpointing disabled.")

train_ds = TextDataset(X_tr_sbert_raw, y_tr, tokenizer, BERT_FT_MAX_LEN)
test_ds = TextDataset(X_te_sbert_raw, y_te, tokenizer, BERT_FT_MAX_LEN)

if RUN_BERT_FINETUNE:
    args = make_training_args(ft_device)
    print(f"Trainer device: {args.device}")
    bert_clf.to(args.device)
    trainer_kwargs = dict(
        model=bert_clf,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in trainer_params:
        trainer_kwargs['processing_class'] = tokenizer
    elif 'tokenizer' in trainer_params:
        trainer_kwargs['tokenizer'] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    t0 = time.time()
    trainer.train()
    print(f"BERT fine-tune training: {time.time()-t0:.1f}s")

    pred_out = trainer.predict(test_ds)
    preds_ft = pred_out.predictions.argmax(axis=1)
    f1_ft = f1_score(y_te, preds_ft, average='macro')
    acc_ft = accuracy_score(y_te, preds_ft)
    print(f"BERT-FT end-to-end: F1={f1_ft:.4f}  Acc={acc_ft:.4f}")
else:
    print("RUN_BERT_FINETUNE=False; extracting embeddings from the base pretrained model.")
    preds_ft = None
    f1_ft = None
    acc_ft = None

# ─────────────────────────────────────────────
# 4. Extract embeddings from the fine-tuned encoder
# ─────────────────────────────────────────────
print("\n[4/6] Extracting embeddings from fine-tuned transformer encoder...")


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


def encode_with_transformer(model, texts, name: str) -> np.ndarray:
    model.eval()
    ds = TextDataset(texts, None, tokenizer, BERT_FT_MAX_LEN)
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=EMB_BATCH_SIZE, shuffle=False, collate_fn=collator)
    device = next(model.parameters()).device
    chunks = []
    t0 = time.time()
    print(f"  Encoding {len(texts):,} {name} texts on {device}...")
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model.base_model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu().numpy().astype(np.float32))
            if step % 200 == 0:
                print(f"    encoded {min(step * EMB_BATCH_SIZE, len(texts)):,}/{len(texts):,}")
    emb = np.vstack(chunks).astype(np.float32)
    print(f"  {name} embeddings: {emb.shape} ({time.time()-t0:.1f}s)")
    return emb


# Use classifier encoder if fine-tuned; otherwise wrap base model in a classifier-like object for consistent code.
if not RUN_BERT_FINETUNE:
    base_encoder = AutoModel.from_pretrained(SBERT_MODEL)
    base_encoder.to(torch.device(DEVICE if DEVICE in ('cuda', 'mps') else 'cpu'))

    class BaseEncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.base_model = encoder

    bert_for_embeddings = BaseEncoderWrapper(base_encoder)
else:
    bert_for_embeddings = bert_clf

X_tr_emb = encode_with_transformer(bert_for_embeddings, X_tr_sbert_raw, 'train')
X_te_emb = encode_with_transformer(bert_for_embeddings, X_te_sbert_raw, 'test')
X_tr_emb = normalize(X_tr_emb).astype(np.float32)
X_te_emb = normalize(X_te_emb).astype(np.float32)
print(f"Fine-tuned embeddings: train={X_tr_emb.shape}, test={X_te_emb.shape}")

clear_accelerator_cache()

# ─────────────────────────────────────────────
# 5. Benchmark SVM / MLP on fine-tuned embeddings
# ─────────────────────────────────────────────
print("\n[5/5] Benchmarking SVM / MLP on fine-tuned BERT embeddings ...")

feature_sets = {
    'bert_ft_emb': (X_tr_emb, X_te_emb),
}

SVM_CS = [0.1, 0.3, 1.0, 3.0, 10.0]


def make_model(kind: str, **kwargs):
    if kind == 'SVM':
        return LinearSVC(
            C=kwargs.get('C', 1.0), max_iter=8000,
            class_weight='balanced',
            random_state=RANDOM_STATE,
        )
    if kind == 'MLP':
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu', solver='adam',
            batch_size=256, max_iter=60, early_stopping=True,
            random_state=RANDOM_STATE,
        )
    raise ValueError(kind)


model_specs = (
    [('SVM', {'C': C}, f'SVM(C={C})') for C in SVM_CS]
    + [('MLP', {}, 'MLP')]
)

results = {}
if preds_ft is not None:
    results['BERT-FT @ end-to-end classifier'] = (f1_ft, acc_ft, preds_ft)

for feat_name, (X_tr, X_te) in feature_sets.items():
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for kind, params, label in model_specs:
        tag = f'{label} @ {feat_name}'
        t0 = time.time()
        clf = make_model(kind, **params)
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        f1 = f1_score(y_te, preds, average='macro')
        acc = accuracy_score(y_te, preds)
        dt = time.time() - t0
        print(f"  {tag:36s} | F1={f1:.4f}  Acc={acc:.4f}  ({dt:.1f}s)")
        results[tag] = (f1, acc, preds)

best_name = max(results, key=lambda k: results[k][0])
f1_best, acc_best, y_pred_best = results[best_name]

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

print("\n=== All Results Summary (F1 macro, Acc) ===")
for name, (f1, acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  F1={f1:.4f}  Acc={acc:.4f}  {name}")
