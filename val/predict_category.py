import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

TRAIN_PATH = DATA_DIR / "queries_train.json"
TEST_PATH = DATA_DIR / "queries_test.json"
SUBMISSION_PATH = OUTPUTS_DIR / "Team_92i_ElMorjen.csv"


def build_query_text(query):
    parts = [query.get("text", ""), query.get("title", "")]
    tags = query.get("tags")
    if tags:
        parts.append(" ".join(tags))
    return " ".join(p for p in parts if p).strip()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    train_queries = load_json(TRAIN_PATH)
    test_queries = load_json(TEST_PATH)

    train_texts = [build_query_text(q) for q in train_queries]
    train_labels = [q["category"] for q in train_queries]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        stop_words="english",
    )
    X_all = vectorizer.fit_transform(train_texts)

    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        multi_class="auto",
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all,
        train_labels,
        test_size=0.2,
        random_state=42,
        stratify=train_labels,
    )
    clf.fit(X_tr, y_tr)
    val_preds = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds, average="macro")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Validation macro F1: {val_f1:.4f}")

    clf.fit(X_all, train_labels)

    test_texts = [build_query_text(q) for q in test_queries]
    X_test = vectorizer.transform(test_texts)
    preds = clf.predict(X_test)

    pred_map = {q["id"]: pred for q, pred in zip(test_queries, preds)}

    submission = pd.read_csv(SUBMISSION_PATH)
    submission["category"] = submission["query_id"].apply(lambda qid: pred_map.get(qid))

    missing_mask = submission["category"].isna()
    if bool(missing_mask.any()):
        missing = submission.loc[missing_mask, "query_id"].head(5).tolist()
        raise ValueError(f"Missing categories for query_id(s): {missing}")

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Updated categories in {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
