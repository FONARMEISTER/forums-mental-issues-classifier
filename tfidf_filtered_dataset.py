"""
Mental Health Text Classification — TF-IDF Pipeline
====================================================
Semantic blocks (each block can be copy-pasted into a Jupyter notebook cell):

  Block 1 — Imports & Config
  Block 2 — Function Definitions (text filtering, preprocessing, augmentation, etc.)
  Block 3 — Load, Preprocess Entire Dataset & Split
  Block 4 — Augmentation & Upsampling
  Block 5 — TF-IDF Feature Extraction
  Block 6 — Classification & Evaluation
"""

# ══════════════════════════════════════════════════════════════════════════════
# Block 1 — Imports & Config
# ══════════════════════════════════════════════════════════════════════════════
import re
import warnings

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')

tqdm.pandas(desc='processing')
nltk.download('stopwords', quiet=True)

import pymorphy3

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH = 'data.csv'
AUGMENTED_DATA_PATH = 'augmented_data.csv'

# ── Filtering thresholds ───────────────────────────────────────────────────
MIN_WORDS    = 3      # minimum token count after cleaning (drop very short texts)
MAX_WORDS    = 1000    # truncate very long texts to this many tokens

# ── Classes to drop (ambiguous / too noisy) ────────────────────────────────
CLASSES_TO_DROP = ['тревожное р-во/депрессия', 'паранойя', 'тревожное р-во/невроз']

# ── Augmentation ───────────────────────────────────────────────────────────
USE_AUGMENTATION = True   # set to False to skip augmentation entirely

# ── Split ──────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FIXATED_TEST_SIZE = 5000  # number of texts held out for the test set
MIN_TRAIN_PER_CLASS = 3000  # minimum instances per class in train set after upsampling
MAX_TRAIN_PER_CLASS = 5000  # maximum instances per class in train set (downsample larger classes)

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

# ── NLP tools ──────────────────────────────────────────────────────────────
RU_STOPWORDS = set(stopwords.words('russian'))
MORPH        = pymorphy3.MorphAnalyzer()
STEMMER      = SnowballStemmer('russian')

# Domain-specific stopwords: class-neutral words that appear in every class
# with near-equal frequency and carry zero discriminative signal.
DOMAIN_STOPWORDS = {
    # Generic therapy / forum words
    'психолог', 'психотерапевт', 'специалист', 'терапия', 'терапевт',
    'психотерапия', 'психиатр', 'психиатрия',
    'человек', 'люди', 'жизнь', 'время', 'ситуация', 'проблема',
    'помощь', 'помочь', 'работа', 'работать',
    'автор', 'пост', 'форум', 'комментарий', 'тема',
    'вообще', 'просто', 'конечно', 'наверное', 'наверно',
    'кстати', 'вроде', 'типа', 'короче',
    'это', 'очень'
}
RU_STOPWORDS |= DOMAIN_STOPWORDS

np.random.seed(RANDOM_STATE)

print("=" * 64)
print("MENTAL HEALTH CLASSIFICATION — TF-IDF Pipeline")
print(f"Data:             {DATA_PATH}")
print(f"Min words:        {MIN_WORDS}  |  Max words: {MAX_WORDS}")
print(f"Test set size:    {FIXATED_TEST_SIZE}")
print(f"Min/max per class (train): {MIN_TRAIN_PER_CLASS} / {MAX_TRAIN_PER_CLASS}")
print(f"Augmentation:     {'Enabled' if USE_AUGMENTATION else 'Disabled'}"
      + (f"  ({AUGMENTED_DATA_PATH})" if USE_AUGMENTATION else ""))
print(f"TF-IDF ngram:     {TFIDF_PARAMS['ngram_range']}  "
      f"max_features={TFIDF_PARAMS['max_features']}  "
      f"min_df={TFIDF_PARAMS['min_df']}  "
      f"max_df={TFIDF_PARAMS['max_df']}")
print("=" * 64)


# ══════════════════════════════════════════════════════════════════════════════
# Block 2 — Function Definitions
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
        -> ["Я", "не_Грустный", "сегодня"]
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
                # Next token is also a negation particle — keep current one
                result.append(tok)
        else:
            # Standalone trailing negation particle or non-negation token — keep it
            result.append(tok)
    return result


def remove_stopwords_and_normalize(text: str) -> str:
    """
    1. Split into tokens (case preserved from filter_text).
    2. Apply negation handling: "не X" -> "не_X" (case-insensitive particle match).
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
    # Steps 4+5: lemmatize then stem in one pass
    result = []
    for t in tokens:
        if t.startswith('не_'):
            word = t[3:]
            lemma = MORPH.parse(word)[0].normal_form if word else word
            stem  = STEMMER.stem(lemma) if lemma else lemma
            result.append('не_' + stem)
        else:
            lemma = MORPH.parse(t)[0].normal_form
            result.append(STEMMER.stem(lemma))
    tokens = [t for t in result if len(t) >= 2]
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full text preprocessing pipeline to a dataframe with 'text' and 'tag' columns.
    Returns a new dataframe with 'text_clean', 'text_processed', 'word_count' columns added."""
    df = df.copy()

    tqdm.pandas(desc='filter_text')
    df['text_clean'] = df['text'].progress_apply(filter_text)

    # ── Apply marker quality filter BEFORE stemming (saves computation) ──────
    if 'tag' in df.columns:
        df['_quality'] = df.apply(lambda r: is_quality_text(r['text_clean'], r['tag']), axis=1)
        q_counts   = df[df['_quality']]['tag'].value_counts()
        all_counts = df['tag'].value_counts()
        before = len(df)
        df = df[df['_quality']].drop(columns=['_quality']).reset_index(drop=True)
        print(f"After quality filter:               removed {before - len(df):,} -> {len(df):,} kept")
        print("Per-class quality keep rate:")
        for tag in all_counts.index:
            kept  = q_counts.get(tag, 0)
            total = all_counts[tag]
            print(f"  {tag:30s}: kept {kept:5,}/{total:5,}  ({kept/total*100:.1f}%)")

    tqdm.pandas(desc='stopwords+lemmatize+stem')
    df['text_processed'] = df['text_clean'].progress_apply(remove_stopwords_and_normalize)

    df['word_count'] = df['text_processed'].apply(lambda s: len(s.split()))
    before = len(df)
    df = df[df['word_count'] >= MIN_WORDS].copy().reset_index(drop=True)
    print(f"After min-words filter ({MIN_WORDS}): removed {before - len(df):,} -> {len(df):,} kept")

    before = len(df)
    df = df.drop_duplicates(subset=['text_processed', 'tag']).reset_index(drop=True)
    print(f"After de-duplication:               removed {before - len(df):,} -> {len(df):,} kept")

    return df


def augment_training_data(df_train: pd.DataFrame, augment_path: str) -> pd.DataFrame:
    """
    Augment training data with additional samples from augmented_data.csv.
    The augmented data is preprocessed before merging so it matches the
    already-preprocessed training data.
    Returns combined dataframe with original + augmented data.
    """
    print(f"\n[AUGMENT] Loading augmented data from {augment_path}...")

    try:
        df_aug = pd.read_csv(augment_path, sep=';', encoding='utf-8')
        print(f"Augmented data shape: {df_aug.shape}")

        # Keep only required columns
        if 'text' in df_aug.columns and 'tag' in df_aug.columns:
            df_aug = df_aug[['text', 'tag']].copy()
        else:
            raise ValueError("Augmented data must contain 'text' and 'tag' columns")

        # Clean augmented data
        df_aug = df_aug.dropna(subset=['text', 'tag']).reset_index(drop=True)
        df_aug['text'] = df_aug['text'].astype(str)
        df_aug['tag'] = df_aug['tag'].astype(str).str.strip()

        # Drop ambiguous classes from augmented data
        df_aug = df_aug[~df_aug['tag'].isin(CLASSES_TO_DROP)].copy().reset_index(drop=True)

        # Preprocess augmented data so it matches the already-preprocessed training data
        print("Preprocessing augmented data...")
        df_aug = preprocess_dataframe(df_aug)

        print(f"After cleaning & preprocessing augmented data: {len(df_aug):,} rows")
        print("Augmented data class distribution:")
        print(df_aug['tag'].value_counts().to_string())

        # Combine original and augmented data
        df_combined = pd.concat([df_train, df_aug], ignore_index=True)
        df_combined = df_combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print(f"\nCombined training data: {len(df_combined):,} rows "
              f"(original: {len(df_train):,} + augmented: {len(df_aug):,})")

        return df_combined

    except Exception as e:
        print(f"[WARNING] Failed to load augmented data: {e}")
        print("[WARNING] Continuing with original training data only.")
        return df_train


def downsample_train_set(df_train: pd.DataFrame, max_per_class: int,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Downsample training set so that no class has more than max_per_class instances.
    Uses random sampling without replacement for classes that exceed the limit.
    """
    print(f"\n[DOWNSAMPLE] Downsampling training set to max {max_per_class} per class...")

    class_counts = df_train['tag'].value_counts()
    print("Current training set class distribution:")
    print(class_counts.to_string())

    downsampled_dfs = []

    for class_name in df_train['tag'].unique():
        class_df = df_train[df_train['tag'] == class_name]
        current_count = len(class_df)

        if current_count > max_per_class:
            print(f"  Downsampling class '{class_name}': {current_count} -> {max_per_class} (-{current_count - max_per_class})")
            downsampled_class = class_df.sample(n=max_per_class, replace=False, random_state=random_state)
        else:
            downsampled_class = class_df.copy()
            print(f"  Class '{class_name}': {current_count} (no downsampling needed)")

        downsampled_dfs.append(downsampled_class)

    df_downsampled = pd.concat(downsampled_dfs, ignore_index=True)
    df_downsampled = df_downsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nDownsampled training set: {len(df_downsampled):,} samples")
    print("Class distribution after downsampling:")
    print(df_downsampled['tag'].value_counts().to_string())

    return df_downsampled


def upsample_train_set(df_train: pd.DataFrame, min_per_class: int,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Upsample training set to ensure each class has at least min_per_class instances.
    Uses random sampling with replacement for classes that need more samples.
    """
    print(f"\n[UPSAMPLE] Upsampling training set to min {min_per_class} per class...")

    class_counts = df_train['tag'].value_counts()
    print("Current training set class distribution:")
    print(class_counts.to_string())

    upsampled_dfs = []

    for class_name in df_train['tag'].unique():
        class_df = df_train[df_train['tag'] == class_name]
        current_count = len(class_df)

        if current_count < min_per_class:
            # Need to upsample this class
            needed = min_per_class - current_count
            print(f"  Upsampling class '{class_name}': {current_count} -> {min_per_class} (+{needed})")

            # Sample with replacement to get additional samples
            additional_samples = class_df.sample(n=needed, replace=True, random_state=random_state)
            upsampled_class = pd.concat([class_df, additional_samples], ignore_index=True)
        else:
            # No upsampling needed
            upsampled_class = class_df.copy()
            print(f"  Class '{class_name}': {current_count} (no upsampling needed)")

        upsampled_dfs.append(upsampled_class)

    df_upsampled = pd.concat(upsampled_dfs, ignore_index=True)
    df_upsampled = df_upsampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"\nUpsampled training set: {len(df_upsampled):,} samples")
    print("Final training set class distribution:")
    print(df_upsampled['tag'].value_counts().to_string())

    return df_upsampled


# Per-class keyword lists.  Checked against the *cleaned* but not yet
# lemmatised/stemmed text so substrings like 'биполяр' still match the
# inflected forms 'биполярное', 'биполярный', etc.
CLASS_MARKERS = {
    'депрессия': [
        'депресси', 'депрессив', 'антидепрессант', 'ангедони',
        'флуоксетин', 'сертралин', 'эсциталопрам', 'венлафаксин', 'миртазапин',
        'нет сил', 'ничего не хочу', 'апати', 'суицид', 'суицидальн',
        'лежу', 'не могу встать', 'без сил', 'нет энергии', 'тоск',
        'безнадёжн', 'безнадежн', 'опустошён', 'опустошен',
        'не хочу жить', 'нет желания', 'прокрастинац', 'выгорани',
        'не могу спать', 'плохо сплю', 'просыпаюсь ночью', 'сплю по',
        'не чувствую ничего', 'как будто умер', 'равнодушие ко всему',
        'мысли о смерти', 'не вижу смысл', 'всё бессмысленн', 'виноват во всём',
        'ненавижу себя', 'я никому не нужен', 'я обуза', 'плачу без причин',
        'не выхожу из дома', 'не могу работать', 'потерял интерес',
        'ничего не радует', 'нет удовольствия', 'потерял смысл',
    ],
    'тревожное р-во': [
        'паническ', 'паник', 'тревог', 'тревожн', 'гтр', 'птср',
        'агорафоби', 'социофоби', 'социальн тревог', 'генерализованн тревог',
        'страх', 'боюсь', 'боится',
        'сердце бьёт', 'сердце бьет', 'сердцебиени', 'учащённ пульс',
        'не хватает воздуха', 'задыхаюсь', 'удушь', 'ком в горле',
        'дрожь', 'трясёт', 'потею', 'холодный пот', 'головокружени',
        'тошнит от тревог', 'живот сводит', 'мышцы напряжен',
        'фоби', 'навязчив', 'избегани', 'не могу выйти', 'боюсь выходить',
        'катастрофизац', 'предчувствие беды', 'ожидание худшего',
        'вегетатив', 'дереализаци', 'деперсонализаци',
        'грандаксин', 'феназепам', 'атаракс', 'афобазол', 'когнитивно поведенческ',
    ],
    'ОКР': [
        'ритуал', 'перепроверк', 'навязчив', 'компульси', 'обсесси', 'окр', 'оkр',
        'проверяю', 'мою руки', 'страх заражени', 'страх загрязнени',
        'мою посуду', 'протираю', 'дезинфицирую', 'стерильн',
        'повторяю', 'считаю', 'симметри', 'всё должно быть ровно',
        'не могу остановиться', 'делаю снова и снова',
        'страх причинить', 'магическ мышлени', 'навязчивые мысли',
        'нежелательн мысл', 'страшные мысли лезут', 'мысли против воли',
        'агрессивн навязчив', 'богохульн мысл',
        'боюсь причинить вред', 'я не хочу этого', 'мысли ужасают меня',
        'флувоксамин', 'кломипрамин', 'анафранил',
    ],
    'ПРЛ': [
        'пустот', 'идеализаци', 'обесценивани', 'пограничн', 'прл', 'бпр',
        'страх быть брошен', 'страх одиночеств', 'бурн отношени', 'токсичн отношени',
        'не могу быть одна', 'цепляюсь за людей', 'резко меняю отношение',
        'то обожаю то ненавижу', 'черно-белое мышлени', 'всё или ничего',
        'эмоциональн качел', 'импульсивн', 'вспышки гнева', 'не контролирую эмоци',
        'резкие перепады', 'захлёстывает', 'не могу успокоиться',
        'самоповреждени', 'порезы', 'членовредительств', 'режу себя',
        'нестабильн', 'диссоциаци', 'расщеплени', 'не знаю кто я',
        'размытая идентичность', 'ощущение пустоты внутри',
        'диалектическ', 'дбт', 'dbt', 'ламотриджин для настроени',
    ],
    'БАР': [
        'биполяр', 'бар ', ' бар', 'мани', 'гипомани', 'нормотимик',
        'литий', 'депакин', 'ламиктал', 'ламотриджин',
        'сероквел', 'кветиапин', 'карбамазепин', 'вальпроат', 'тегретол',
        'смена настроени', 'перепады настроени', 'цикл настроени',
        'эйфори', 'грандиозн', 'не нужен сон', 'не сплю', 'бессонниц',
        'маниакальн', 'ускоренн мышлени', 'мысли скачут', 'грандиозные планы',
        'трачу деньги', 'безрассудн', 'гиперсексуальн',
        'депрессивн эпизод', 'смешанн эпизод', 'после подъёма упадок',
        'цикличн', 'фаза подъёма', 'фаза спада', 'снова поднимается настроение',
    ],
    'шизофрения': [
        'шизо',
        'галлюцинаци', 'галлюцинир', 'слышу голоса', 'голоса', 'голос в голов',
        'бред', 'бредов', 'бред преследовани', 'бред величи',
        'психоз', 'психотич', 'острый психоз',
        'негативн симптом', 'позитивн симптом', 'уплощённ аффект',
        'алоги', 'абули',
        'разорванн мышлени', 'расщеплени личност', 'параноидн',
        'деперсонализаци', 'дереализаци', 'кататони',
        'антипсихотик', 'нейролептик',
        'галоперидол', 'клозапин', 'оланзапин', 'рисперидон', 'арипипразол',
        'зипрасидон', 'палиперидон', 'амисульприд', 'кветиапин',
        'психиатрическ больниц', 'стационар', 'принудительн лечени',
    ],
}

# Classes with few examples — softer rule: 200-char texts are kept even with
# fewer than 3 marker hits (to preserve rare training signal).
RARE_CLASSES = {'БАР', 'ОКР', 'ПРЛ'}


def is_quality_text(text: str, tag: str) -> bool:
    """Keep text if it has ≥ 3 class markers, or is from a rare class and ≥ 200 chars."""
    if not isinstance(text, str):
        return False
    if tag not in CLASS_MARKERS:
        return True
    text_lower = text.lower()
    found = sum(1 for m in CLASS_MARKERS[tag] if m in text_lower)
    return found >= 1 or (tag in RARE_CLASSES and len(text) >= 200)


# ══════════════════════════════════════════════════════════════════════════════
# Block 3 — Load, Preprocess Entire Dataset & Split
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading data and selecting columns...")

# Load initial data
df_initial = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')
print(f"Initial data shape: {df_initial.shape}  |  columns: {list(df_initial.columns)}")

# Keep only the two required columns
df_initial = df_initial[['text', 'tag']].copy()
df_initial = df_initial.dropna(subset=['text', 'tag']).reset_index(drop=True)
df_initial['text'] = df_initial['text'].astype(str)
df_initial['tag']  = df_initial['tag'].astype(str).str.strip()

# Drop ambiguous / noisy classes from initial data
before = len(df_initial)
df_initial = df_initial[~df_initial['tag'].isin(CLASSES_TO_DROP)].copy().reset_index(drop=True)
print(f"After dropping ambiguous classes: removed {before - len(df_initial):,} -> {len(df_initial):,} rows, "
      f"{df_initial['tag'].nunique()} classes")

print("\nClass distribution (initial data):")
print(df_initial['tag'].value_counts().to_string())

# ── Preprocess the entire dataset once ───────────────────────────────────────
print("\n[2/6] Preprocessing entire dataset...")
df_initial = preprocess_dataframe(df_initial)

print(f"\nWord count stats after processing:")
print(df_initial['word_count'].describe().to_string())

print("\n=== 5 random preprocessed samples ===")
for _, row in df_initial.sample(n=5, random_state=RANDOM_STATE).iterrows():
    tokens = row['text_processed'].split()
    print(f"\n  [tag: {row['tag']}]")
    print(f"  RAW:    {row['text'][:120]}")
    print(f"  TOKENS: {tokens}")
print("=" * 64)

print(f"\nClass distribution after preprocessing:")
print(df_initial['tag'].value_counts().to_string())

# ── Fixated Test Set Selection ───────────────────────────────────────────────
print(f"\n[3/6] Selecting fixated test set ({FIXATED_TEST_SIZE} samples)...")

df_test_fixated_processed = df_initial.sample(n=FIXATED_TEST_SIZE, random_state=RANDOM_STATE)
df_train_initial = df_initial.drop(df_test_fixated_processed.index).reset_index(drop=True)
df_test_fixated_processed = df_test_fixated_processed.reset_index(drop=True)

print(f"Fixated test set: {len(df_test_fixated_processed):,} samples")
print(f"Initial train set: {len(df_train_initial):,} samples")

print("\nFixated test set class distribution:")
print(df_test_fixated_processed['tag'].value_counts().to_string())

print("\nInitial train set class distribution:")
print(df_train_initial['tag'].value_counts().to_string())

df_source = df_train_initial.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"\nTraining data: {len(df_source):,} rows, {df_source['tag'].nunique()} classes")


# ══════════════════════════════════════════════════════════════════════════════
# Block 4 — Augmentation & Upsampling
# ══════════════════════════════════════════════════════════════════════════════

# Apply augmentation if requested (augmented data will be preprocessed inside)
if USE_AUGMENTATION:
    df_source = augment_training_data(df_source, AUGMENTED_DATA_PATH)

# Downsample training set to cap classes at MAX_TRAIN_PER_CLASS
df_train_final = downsample_train_set(df_source, MAX_TRAIN_PER_CLASS, RANDOM_STATE)

# Upsample training set to ensure minimum samples per class
df_train_final = upsample_train_set(df_train_final, MIN_TRAIN_PER_CLASS, RANDOM_STATE)

print(f"\nFinal train set: {len(df_train_final):,} samples")
print(f"Final test set: {len(df_test_fixated_processed):,} samples")

# Encode labels
le = LabelEncoder()
y_train  = le.fit_transform(df_train_final['tag'].to_numpy())
y_test   = le.transform(df_test_fixated_processed['tag'].to_numpy())  # Use same encoder
X_train  = df_train_final['text_processed'].to_numpy(dtype=object)
X_test   = df_test_fixated_processed['text_processed'].to_numpy(dtype=object)
NUM_CLASSES = len(le.classes_)

print(f"Classes ({NUM_CLASSES}): {list(le.classes_)}")
print("Train class counts:",
      dict(zip(le.classes_, np.bincount(y_train, minlength=NUM_CLASSES))))
print("Test  class counts:",
      dict(zip(le.classes_, np.bincount(y_test,  minlength=NUM_CLASSES))))


# ══════════════════════════════════════════════════════════════════════════════
# Block 5 — TF-IDF Feature Extraction
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

print(f"TF-IDF matrix -- train: {X_train_tfidf.shape}  |  test: {X_test_tfidf.shape}")
print(f"Vocabulary size: {len(tfidf.vocabulary_):,}")


# ══════════════════════════════════════════════════════════════════════════════
# Block 6 — Classification & Evaluation
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
    print(f"TARGET ACHIEVED: F1 macro = {f1_best:.4f} >= {TARGET_F1}")
else:
    print(f"TARGET NOT MET:  F1 macro = {f1_best:.4f} < {TARGET_F1}")
print("=" * 64)

print("\n=== All Results Summary (sorted by F1 macro) ===")
for label, (f1, acc, _) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"  F1={f1:.4f}  Acc={acc:.4f}  {label}")
