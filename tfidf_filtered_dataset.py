"""
Mental Health Text Classification — TF-IDF Pipeline
====================================================
Semantic blocks (each block can be copy-pasted into a Jupyter notebook cell):

  Block 1 — Imports & Config
  Block 2 — Load & Select Columns
  Block 3 — Text Filtering (noise reduction)
  Block 4 — Stopword Removal, Lemmatization & Stemming
  Block 5 — Train / Test Split (80/20 stratified)
  Block 6 — TF-IDF Feature Extraction
  Block 7 — Classification & Evaluation
  Block 8 — Iterative Dataset Pruning (greedy noise removal)
"""

# ══════════════════════════════════════════════════════════════════════════════
# Block 1 — Imports & Config
# ══════════════════════════════════════════════════════════════════════════════
import re
import warnings
import os

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

tqdm.pandas(desc='processing')
nltk.download('stopwords', quiet=True)

import pymorphy3 as pymorphy2

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = 'data.csv'
PRUNING_CHECKPOINT_PATH = 'data_filtered_pruned.csv'

# ── Filtering thresholds ───────────────────────────────────────────────────
MIN_WORDS    = 7      # minimum token count after cleaning (drop very short texts)
MAX_WORDS    = 1000    # truncate very long texts to this many tokens
MIN_CHAR_LEN = 10     # drop texts shorter than this many characters (raw)

# ── Classes to drop (ambiguous / too noisy) ────────────────────────────────
CLASSES_TO_DROP = ['тревожное р-во/депрессия', 'паранойя', 'тревожное р-во/невроз']

# ── Split ──────────────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── TF-IDF — change these parameters to tune noise reduction ──────────────
TFIDF_PARAMS = dict(
    analyzer = 'word',
    ngram_range = (1, 2),      # unigrams + bigrams
    max_features= 50000,     # vocabulary cap
    sublinear_tf= True,        # apply 1 + log(tf) scaling
    min_df      = 3,           # ignore terms appearing in fewer than N docs
    max_df      = 0.85,        # ignore terms appearing in more than 95 % of docs
    token_pattern= r'[а-яёa-zА-ЯA-Z]{2,}',  # only Cyrillic/Latin words ≥ 2 chars
)

# ── Pruning config ─────────────────────────────────────────────────────────
PRUNING_MIN_TOTAL   = 25_000   # stop when fewer than this many texts remain
PRUNING_MIN_PER_CLASS = 500    # each non-dropped class must keep at least this many texts

# ── NLP tools ──────────────────────────────────────────────────────────────
RU_STOPWORDS = set(stopwords.words('russian'))
MORPH        = pymorphy2.MorphAnalyzer()
STEMMER      = SnowballStemmer('russian')

# Domain-specific stopwords: class-neutral words that appear in every class
# with near-equal frequency and carry zero discriminative signal.
DOMAIN_STOPWORDS = {
    # Generic therapy / forum words
    'психолог', 'психотерапевт', 'специалист', 'терапия', 'терапевт',
    'психотерапия', 'психиатр', 'психиатрия',
    'человек', 'люди', 'жизнь', 'время', 'ситуация', 'проблема',
    'помощь', 'помочь', 'работа', 'работать',
    # Forum meta-words
    'автор', 'пост', 'форум', 'комментарий', 'тема',
    # Filler discourse markers
    'вообще', 'просто', 'конечно', 'наверное', 'наверно',
    'кстати', 'вроде', 'типа', 'короче',
}
RU_STOPWORDS |= DOMAIN_STOPWORDS

np.random.seed(RANDOM_STATE)

print("=" * 64)
print("MENTAL HEALTH CLASSIFICATION — TF-IDF Pipeline")
print(f"Data:        {DATA_PATH}")
print(f"Min words:   {MIN_WORDS}  |  Max words: {MAX_WORDS}")
print(f"Test size:   {TEST_SIZE}")
print(f"TF-IDF ngram:{TFIDF_PARAMS['ngram_range']}  "
      f"max_features={TFIDF_PARAMS['max_features']}  "
      f"min_df={TFIDF_PARAMS['min_df']}  "
      f"max_df={TFIDF_PARAMS['max_df']}")
print(f"Pruning min total:     {PRUNING_MIN_TOTAL:,}")
print(f"Pruning min per class: {PRUNING_MIN_PER_CLASS:,}")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# Block 2 — Load & Select Columns
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading data and selecting columns...")

# Resume from checkpoint if it exists
if os.path.exists(PRUNING_CHECKPOINT_PATH):
    print(f"[RESUME] Found pruning checkpoint: {PRUNING_CHECKPOINT_PATH}")
    df_source = pd.read_csv(PRUNING_CHECKPOINT_PATH, sep=';', encoding='utf-8')
    print(f"Checkpoint shape: {df_source.shape}")
else:
    df_source = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')
    print(f"Raw shape: {df_source.shape}  |  columns: {list(df_source.columns)}")

# Keep only the two required columns
df_source = df_source[['text', 'tag']].copy()
df_source = df_source.dropna(subset=['text', 'tag']).reset_index(drop=True)
df_source['text'] = df_source['text'].astype(str)
df_source['tag']  = df_source['tag'].astype(str).str.strip()
df_source = df_source.sample(frac = 1, random_state = RANDOM_STATE).reset_index(drop=True)

print(f"After column selection & dropna: {len(df_source):,} rows, {df_source['tag'].nunique()} classes")

# Drop ambiguous / noisy classes
before = len(df_source)
df_source = df_source[~df_source['tag'].isin(CLASSES_TO_DROP)].copy().reset_index(drop=True)
print(f"After dropping ambiguous classes: removed {before - len(df_source):,} → {len(df_source):,} rows, "
      f"{df_source['tag'].nunique()} classes")

print("\nClass distribution (raw):")
print(df_source['tag'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Block 3 — Text Filtering (noise reduction)
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compiled regex patterns
_URL_RE    = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_RE  = re.compile(r'\S+@\S+\.\S+')
_HTML_RE   = re.compile(r'<[^>]+>')
_MENTION_RE= re.compile(r'@\w+')           # @username mentions
_HASHTAG_RE= re.compile(r'#\w+')           # #hashtags
_DIGIT_RE  = re.compile(r'\b\d+\b')        # standalone numbers
_PUNCT_RE  = re.compile(r'[^а-яёА-ЯЁa-zA-Z\s]')  # keep only letters + spaces
_WS_RE     = re.compile(r'\s+')
# Collapse 3+ repeated characters to 2: "оооочень" → "оочень"
_REPEAT_RE = re.compile(r'(.)\1{2,}')

# ── Domain noise patterns (strip substring, not the whole row) ──────────────
# Greeting openers: "Здравствуйте!", "Добрый день,", "Доброго времени суток" etc.
_GREETING_RE = re.compile(
    r'^\s*(?:здравствуйте[!,.]?\s*|добрый\s+\w+[!,.]?\s*'
    r'|доброго\s+\w+(?:\s+\w+)?[!,.]?\s*|привет[!,.]?\s*)+',
    re.I,
)
# Sign-off phrases at the end of text
_SIGNOFF_RE = re.compile(
    r'(?:спасибо\s+за\s+(?:ваш\s+)?ответ|с\s+уважением[\s\w,]*'
    r'|всего\s+доброго|удачи\s+вам|хорошего\s+дня)[.!]?\s*$',
    re.I,
)
# "Автор," meta-address at the start (forum convention)
_META_AUTHOR_RE = re.compile(r'^\s*автор\s*,\s*', re.I)
# Book / therapy recommendation boilerplate
_BOOK_RE = re.compile(
    r'(?:рекомендую\s+(?:книгу|почитать)|книга\s+[«\"][^»\"]{1,60}[»\"]'
    r'|читайте\s+книгу)',
    re.I,
)
# Social media / channel promos
_PROMO_RE = re.compile(
    r'(?:подписывайтесь(?:\s+на\s+(?:мой|наш)\s+\w+)?'
    r'|мой\s+(?:канал|блог|телеграм|telegram)'
    r'|t\.me/\S+|instagram\.com/\S+|vk\.com/\S+)',
    re.I,
)

# Negation particles that flip the meaning of the following word
_NEGATION_PARTICLES = {'не', 'ни'}


def filter_text(text: str) -> str:
    """
    Multi-step noise reduction pipeline.
    Returns a case-preserved, cleaned string (stopwords still present at this stage).
    Case is preserved so that pymorphy3 can use capitalisation as a morphological cue.

    Steps:
      1. Strip URLs, emails, HTML, @mentions, #hashtags, standalone digits.
      2. Strip domain noise: greetings, sign-offs, "Автор,", book refs, promos.
      3. Remove non-letter characters (keep Cyrillic, Latin, spaces).
      4. Collapse 3+ repeated chars to 2.
      5. Truncate to MAX_WORDS tokens.
    """
    # Step 1 — technical noise
    text = _URL_RE.sub(' ', text)
    text = _EMAIL_RE.sub(' ', text)
    text = _HTML_RE.sub(' ', text)
    text = _MENTION_RE.sub(' ', text)
    text = _HASHTAG_RE.sub(' ', text)
    text = _DIGIT_RE.sub(' ', text)
    # Step 2 — domain noise (strip substrings, preserve the rest)
    text = _GREETING_RE.sub(' ', text)
    text = _SIGNOFF_RE.sub(' ', text)
    text = _META_AUTHOR_RE.sub(' ', text)
    text = _BOOK_RE.sub(' ', text)
    text = _PROMO_RE.sub(' ', text)
    # Step 3 — keep only letters + spaces
    text = _PUNCT_RE.sub(' ', text)
    # Step 4 — collapse repeated chars
    text = _REPEAT_RE.sub(r'\1\1', text)
    text = _WS_RE.sub(' ', text).strip()
    # Step 5 — truncate
    tokens = text.split()
    if len(tokens) > MAX_WORDS:
        tokens = tokens[:MAX_WORDS]
    return ' '.join(tokens)


def _apply_negation(tokens: list[str]) -> list[str]:
    """
    Prefix the token immediately following a negation particle with 'не_'.
    Comparison against negation particles is case-insensitive so "Не", "НЕ" etc. are caught.
    The particle itself is discarded (it would be removed as a stopword anyway).

    Example:
        ["Я", "не", "Грустный", "сегодня"]
        → ["Я", "не_Грустный", "сегодня"]
    """
    result = []
    skip_next = False
    for i, tok in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        tok_lower = tok.lower()
        if tok_lower in _NEGATION_PARTICLES and i + 1 < len(tokens):
            next_tok = tokens[i + 1]
            # Only prefix if the next token is not itself a negation particle
            if next_tok.lower() not in _NEGATION_PARTICLES:
                result.append('не_' + next_tok)   # preserve case of the content word
                skip_next = True
        else:
            result.append(tok)
    return result


def remove_stopwords_and_normalize(text: str) -> str:
    """
    1. Split into tokens (case preserved from filter_text).
    2. Apply negation handling: "не X" → "не_X" (case-insensitive particle match).
    3. Drop Russian stopwords (case-insensitive comparison; negated tokens are kept).
    4. Lemmatize each token with pymorphy3 — returns lowercase normal form.
    5. Stem the lemma with Snowball — reduces inflectional variants further.
    6. Drop tokens shorter than 2 characters.

    After step 4 all tokens are lowercase (pymorphy3 normal_form is always lowercase),
    so case-sensitivity is preserved through steps 1-3 and normalised from step 4 onward.
    """
    tokens = text.split()
    # Step 2: negation handling (before stopword removal so "не" is still present)
    tokens = _apply_negation(tokens)
    # Step 3: drop stopwords — compare lowercase so "Я", "Мне" etc. are caught
    tokens = [t for t in tokens if t.lower() not in RU_STOPWORDS]
    # Step 4: lemmatize — handle "не_word" compounds; pymorphy3 returns lowercase
    lemmatized = []
    for t in tokens:
        if t.startswith('не_'):
            word = t[3:]
            lemma = MORPH.parse(word)[0].normal_form if word else word
            lemmatized.append('не_' + lemma)
        else:
            lemmatized.append(MORPH.parse(t)[0].normal_form)
    tokens = lemmatized
    # Step 5: stem the lemma (negated prefix preserved, only word part stemmed)
    stemmed = []
    for t in tokens:
        if t.startswith('не_'):
            word = t[3:]
            stemmed.append('не_' + STEMMER.stem(word) if word else t)
        else:
            stemmed.append(STEMMER.stem(t))
    tokens = stemmed
    tokens = [t for t in tokens if len(t) >= 2]
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply full text preprocessing pipeline to a dataframe with 'text' and 'tag' columns.
    Returns a new dataframe with 'text_clean', 'text_processed', 'word_count' columns added."""
    df = df.copy()

    if verbose:
        tqdm.pandas(desc='filter_text')
        df['text_clean'] = df['text'].progress_apply(filter_text)
    else:
        df['text_clean'] = df['text'].apply(filter_text)

    before = len(df)
    df = df[df['text_clean'].str.len() >= MIN_CHAR_LEN].copy().reset_index(drop=True)
    if verbose:
        print(f"After min-char filter ({MIN_CHAR_LEN}): removed {before - len(df):,} → {len(df):,} kept")

    if verbose:
        tqdm.pandas(desc='stopwords+lemmatize+stem')
        df['text_processed'] = df['text_clean'].progress_apply(remove_stopwords_and_normalize)
    else:
        df['text_processed'] = df['text_clean'].apply(remove_stopwords_and_normalize)

    df['word_count'] = df['text_processed'].apply(lambda s: len(s.split()))
    before = len(df)
    df = df[df['word_count'] >= MIN_WORDS].copy().reset_index(drop=True)
    if verbose:
        print(f"After min-words filter ({MIN_WORDS}): removed {before - len(df):,} → {len(df):,} kept")

    before = len(df)
    df = df.drop_duplicates(subset=['text_processed', 'tag']).reset_index(drop=True)
    if verbose:
        print(f"After de-duplication:               removed {before - len(df):,} → {len(df):,} kept")

    return df


def run_pipeline(df_processed: pd.DataFrame, verbose: bool = True) -> float:
    """
    Given a preprocessed dataframe (with 'text_processed' and 'tag' columns),
    run the full TF-IDF + classifier pipeline and return the best macro-F1.
    """
    le = LabelEncoder()
    y  = le.fit_transform(df_processed['tag'].to_numpy())
    X  = df_processed['text_processed'].to_numpy(dtype=object)
    num_classes = len(le.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y,
    )

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf  = tfidf.transform(X_test)

    SVC_CS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

    model_specs = [
        (f'LinearSVC  C={C}',
         LinearSVC(C=C, max_iter=8000, class_weight='balanced',
                   random_state=RANDOM_STATE))
        for C in SVC_CS
    ]

    best_f1 = 0.0
    for label, clf in model_specs:
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        f1 = f1_score(y_test, y_pred, average='macro')
        if verbose:
            acc = accuracy_score(y_test, y_pred)
            print(f"  {label:<35s} | F1={f1:.4f}  Acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1

    return best_f1


# ══════════════════════════════════════════════════════════════════════════════
# Block 3 — Text Filtering (noise reduction)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Applying text filters...")
df = preprocess_dataframe(df_source, verbose=True)

print(f"\nWord count stats after processing:")
print(df['word_count'].describe().to_string())

print("\n=== 5 random preprocessed samples ===")
for _, row in df.sample(n=5, random_state=RANDOM_STATE).iterrows():
    tokens = row['text_processed'].split()
    print(f"\n  [tag: {row['tag']}]")
    print(f"  RAW:    {row['text'][:120]}")
    print(f"  TOKENS: {tokens}")
print("=" * 64)

print(f"\nClass distribution after filtering:")
print(df['tag'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# Block 5 — Train / Test Split (80 / 20, stratified)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Splitting into train / test (80/20 stratified)...")

le = LabelEncoder()
y  = le.fit_transform(df['tag'].to_numpy())
X  = df['text_processed'].to_numpy(dtype=object)
NUM_CLASSES = len(le.classes_)

print(f"Classes ({NUM_CLASSES}): {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,
)

print(f"Train: {len(y_train):,}  |  Test: {len(y_test):,}")
print("Train class counts:",
      dict(zip(le.classes_, np.bincount(y_train, minlength=NUM_CLASSES))))
print("Test  class counts:",
      dict(zip(le.classes_, np.bincount(y_test,  minlength=NUM_CLASSES))))


# ══════════════════════════════════════════════════════════════════════════════
# Block 6 — TF-IDF Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Extracting TF-IDF features...")

tfidf = TfidfVectorizer(**TFIDF_PARAMS)

with tqdm(total=2, desc='TF-IDF', unit='step',
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}') as pbar:
    pbar.set_description('TF-IDF fit_transform (train)')
    X_train_tfidf = tfidf.fit_transform(X_train)
    pbar.set_postfix(shape=str(X_train_tfidf.shape))
    pbar.update(1)

    pbar.set_description('TF-IDF transform (test)')
    X_test_tfidf = tfidf.transform(X_test)
    pbar.set_postfix(shape=str(X_test_tfidf.shape))
    pbar.update(1)

print(f"TF-IDF matrix — train: {X_train_tfidf.shape}  |  test: {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")


# ══════════════════════════════════════════════════════════════════════════════
# Block 7 — Classification & Evaluation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Training classifiers and evaluating...")

TARGET_F1 = 0.60

results = {}  # label -> (f1_macro, accuracy, y_pred)

SVC_CS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]

# ── Helper for sklearn classifiers ─────────────────────────────────────────
def evaluate(label: str, clf, pbar: tqdm) -> None:
    """Fit clf on train TF-IDF, predict on test, store and print metrics."""
    pbar.set_description(f'fitting  {label}')
    clf.fit(X_train_tfidf, y_train)
    pbar.set_description(f'predict  {label}')
    y_pred = clf.predict(X_test_tfidf)
    f1     = f1_score(y_test, y_pred, average='macro')
    acc    = accuracy_score(y_test, y_pred)
    results[label] = (f1, acc, y_pred)
    pbar.set_postfix(F1=f'{f1:.4f}', Acc=f'{acc:.4f}')
    pbar.update(1)
    print(f"  {label:<35s} | F1={f1:.4f}  Acc={acc:.4f}")


# ── LinearSVC sweep ────────────────────────────────────────────────────────
svc_specs = [
    (f'LinearSVC  C={C}',
     LinearSVC(C=C, max_iter=8000, class_weight='balanced',
               random_state=RANDOM_STATE))
    for C in SVC_CS
]

with tqdm(total=len(svc_specs), desc='LinearSVC', unit='model',
          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}') as pbar:
    for label, clf in svc_specs:
        evaluate(label, clf, pbar)

# ── Best model report ──────────────────────────────────────────────────────
best_label = max(results, key=lambda k: results[k][0])
f1_best, acc_best, y_pred_best = results[best_label]

print("\n" + "=" * 64)
print(f"BEST MODEL: {best_label}")
print("=" * 64)
print(f"Accuracy:      {acc_best:.4f}  ({acc_best * 100:.2f}%)")
print(f"F1 (macro):    {f1_best:.4f}")
print(f"F1 (weighted): {f1_score(y_test, y_pred_best, average='weighted'):.4f}")
print()
print("=== Per-class report ===")
print(classification_report(
    y_test, y_pred_best,
    target_names=list(le.classes_),
    digits=4,
))

print("=" * 64)
if f1_best >= TARGET_F1:
    print(f"✅ TARGET ACHIEVED: F1 macro = {f1_best:.4f} >= {TARGET_F1}")
else:
    print(f"❌ TARGET NOT MET:  F1 macro = {f1_best:.4f} < {TARGET_F1}")
print("=" * 64)

print("\n=== All Results Summary (sorted by F1 macro) ===")
for label, (f1, acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  F1={f1:.4f}  Acc={acc:.4f}  {label}")


# ══════════════════════════════════════════════════════════════════════════════
# Block 8 — Iterative Dataset Pruning (greedy noise removal)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 64)
print("BLOCK 8 — Iterative Dataset Pruning")
print(f"  Stop when total texts < {PRUNING_MIN_TOTAL:,}")
print(f"  Min texts per class:    {PRUNING_MIN_PER_CLASS:,}")
print("=" * 64)

# Preprocessing was already done once in Block 3 (df).
# Pruning works entirely on the already-preprocessed data — no re-preprocessing per iteration.
# An '_id' column links each preprocessed row back to its original raw row in df_source,
# so checkpoints can be saved in the original text format.

# Assign a stable id to each source row, then carry it through preprocessing manually
# (preprocess_dataframe resets the index, so we inline the steps to preserve _id).
df_source_with_id = df_source.copy()
df_source_with_id['_id'] = np.arange(len(df_source_with_id))

print("\n[Block 8] Building id-tracked preprocessed frame (one-time)...")
_df_tmp = df_source_with_id.copy()
_df_tmp['text_clean'] = _df_tmp['text'].apply(filter_text)
_df_tmp = _df_tmp[_df_tmp['text_clean'].str.len() >= MIN_CHAR_LEN].copy().reset_index(drop=True)
_df_tmp['text_processed'] = _df_tmp['text_clean'].apply(remove_stopwords_and_normalize)
_df_tmp['word_count'] = _df_tmp['text_processed'].apply(lambda s: len(s.split()))
_df_tmp = _df_tmp[_df_tmp['word_count'] >= MIN_WORDS].copy().reset_index(drop=True)
_df_tmp = _df_tmp.drop_duplicates(subset=['text_processed', 'tag']).reset_index(drop=True)
print(f"  id-tracked frame: {len(_df_tmp):,} rows")

# Working frame for pruning: only the columns needed by run_pipeline + _id for checkpoints.
df_proc_current = _df_tmp[['text_processed', 'tag', '_id']].copy().reset_index(drop=True)

best_f1_pruning = f1_best       # baseline from the full dataset run above
iteration = 0


def _save_checkpoint(df_proc: pd.DataFrame, path: str, iteration: int) -> None:
    """Save raw text+tag for surviving rows back to disk."""
    surviving_ids = df_proc['_id'].values
    df_raw_save = df_source_with_id[df_source_with_id['_id'].isin(surviving_ids)][['text', 'tag']]
    df_raw_save.to_csv(path, sep=';', index=False, encoding='utf-8')
    print(f"  [checkpoint] saved {len(df_raw_save):,} rows → {path}  (iter {iteration})")


print(f"\nBaseline F1 (macro): {best_f1_pruning:.4f}  |  Dataset size: {len(df_proc_current):,}")

# Sequential (brute-force) traversal:
#   - Walk rows 0, 1, 2, … in order; wrap around at the end.
#   - Stop when a full pass over the current dataset produces zero removals.
cursor = 0          # current position in df_proc_current
pass_removals = 0   # removals accepted in the current pass
rows_tried_this_pass = 0  # rows examined since last accepted removal or pass start

while len(df_proc_current) >= PRUNING_MIN_TOTAL:
    iteration += 1

    n = len(df_proc_current)

    # Advance cursor (wrap around)
    remove_pos = cursor % n
    cursor = (cursor + 1) % n

    tag_removed = df_proc_current.at[remove_pos, 'tag']

    # Check per-class eligibility
    class_count = (df_proc_current['tag'] == tag_removed).sum()
    if class_count <= PRUNING_MIN_PER_CLASS:
        rows_tried_this_pass += 1
        # Check if we've gone through the whole dataset without a removal
        if rows_tried_this_pass >= n:
            print(f"[iter {iteration}] Full pass with 0 removals. Stopping.")
            break
        continue

    # Build trial subset — no re-preprocessing needed
    df_trial = df_proc_current.drop(index=remove_pos).reset_index(drop=True)

    if len(df_trial) < PRUNING_MIN_TOTAL:
        print(f"[iter {iteration}] Removing one row would drop below {PRUNING_MIN_TOTAL:,}. Stopping.")
        break

    # Run pipeline on trial subset (silent)
    trial_f1 = run_pipeline(df_trial, verbose=False)

    improved = trial_f1 > best_f1_pruning
    print(f"[iter {iteration:4d}] pos={remove_pos}  size={len(df_trial):,}  "
          f"trial_F1={trial_f1:.4f}  best_F1={best_f1_pruning:.4f}  "
          f"{'✅ IMPROVED' if improved else '❌ no gain'}  "
          f"tag={tag_removed!r}", flush=True)

    if improved:
        best_f1_pruning = trial_f1
        df_proc_current = df_trial
        pass_removals += 1
        rows_tried_this_pass = 0  # reset: new dataset, fresh pass counter
        # cursor already advanced; after reset_index the next row is at the same position
        # (rows shifted up), so cursor stays correct — no adjustment needed.
        _save_checkpoint(df_proc_current, PRUNING_CHECKPOINT_PATH, iteration)
    else:
        rows_tried_this_pass += 1
        # Check if we've gone through the whole dataset without a removal
        if rows_tried_this_pass >= n:
            print(f"[iter {iteration}] Full pass with 0 removals. Stopping.")
            break

print("\n" + "=" * 64)
print(f"Pruning finished after {iteration} iterations.")
print(f"Final dataset size: {len(df_proc_current):,}")
print(f"Best F1 (macro):    {best_f1_pruning:.4f}  (baseline was {f1_best:.4f})")
print("=" * 64)

# Final class distribution
print("\nFinal class distribution:")
print(df_proc_current['tag'].value_counts().to_string())
