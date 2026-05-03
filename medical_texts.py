"""
Fetches medical texts from HuggingFace and builds a long corpus for needle-in-a-haystack
long context evaluation. Needles are specific clinical facts planted at fixed percentage
positions throughout the corpus.
"""

import os
from typing import List, Tuple

CHARS_PER_TOKEN = 4  # rough estimate for English medical text

# Each needle is a specific clinical fact planted at a given depth in the corpus.
# Questions require finding and extracting the exact planted fact.
NEEDLES = [
    {
        "id": 1,
        "position_percent": 5,
        "text": (
            "CLINICAL CASE REPORT (MRN-78234): John Anderson, 67-year-old male, "
            "presented to the ICU on 14-March-2025 with septic shock secondary to "
            "gram-negative bacteraemia. Admission SOFA score was 12 points. "
            "He was initiated on norepinephrine at 0.15 mcg/kg/min and broad-spectrum "
            "antibiotics (piperacillin-tazobactam plus vancomycin)."
        ),
        "question": "What was the SOFA score of patient John Anderson (MRN-78234) on admission to the ICU?",
        "expected_answer": "12",
    },
    {
        "id": 2,
        "position_percent": 25,
        "text": (
            "In the CARDIO-447 randomized controlled trial evaluating novel anticoagulation "
            "strategies in high-risk atrial fibrillation patients, treatment arm B "
            "(factor Xa inhibitor plus antiplatelet therapy) demonstrated a 90-day all-cause "
            "mortality rate of 23.4%, compared to 31.2% in the control arm receiving standard "
            "anticoagulation (p=0.003, NNT=13). The trial enrolled 2,847 patients across 47 centres."
        ),
        "question": "What was the 90-day mortality rate in treatment arm B of the CARDIO-447 trial?",
        "expected_answer": "23.4%",
    },
    {
        "id": 3,
        "position_percent": 50,
        "text": (
            "The Oslo Cardiac Registry (2019-2023) encompassing 4,847 consecutive patients "
            "admitted for acute coronary syndromes reported a median hospital length of stay "
            "of 14.2 days with an interquartile range of 8 to 21 days. Factors independently "
            "associated with prolonged hospitalisation included diabetes mellitus (OR 1.87), "
            "chronic renal dysfunction (OR 2.34), and in-hospital procedural complications (OR 3.12)."
        ),
        "question": "What was the median hospital length of stay and IQR reported by the Oslo Cardiac Registry?",
        "expected_answer": "14.2 days, IQR 8-21 days",
    },
    {
        "id": 4,
        "position_percent": 75,
        "text": (
            "According to Dr. Sarah Chen's validated sepsis resuscitation protocol published in "
            "Critical Care Medicine (2024), vasopressor titration decisions must be guided by a "
            "mean arterial pressure target of greater than 65 mmHg. Deviation below this threshold "
            "for more than 15 consecutive minutes triggers automatic reassessment and escalation "
            "to second-line vasopressors."
        ),
        "question": "According to Dr. Sarah Chen's sepsis protocol, what MAP threshold guides vasopressor titration decisions?",
        "expected_answer": "MAP > 65 mmHg",
    },
    {
        "id": 5,
        "position_percent": 95,
        "text": (
            "The RESPIRE-2024 multi-centre randomised trial (n=892) evaluating combination "
            "antimicrobial therapy in severe community-acquired pneumonia found that the "
            "AZITHRO-DEXA-IV protocol (azithromycin 500mg plus dexamethasone 10mg intravenously "
            "daily) achieved a clinical response rate of 87.3% at day 7, significantly "
            "outperforming standard beta-lactam monotherapy (68.1%, p<0.001, OR 3.21)."
        ),
        "question": "What clinical response rate at day 7 did the AZITHRO-DEXA-IV protocol achieve in the RESPIRE-2024 trial?",
        "expected_answer": "87.3%",
    },
]


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def fetch_medical_corpus(target_tokens: int, max_articles: int = 10000) -> List[str]:
    """
    Stream medical articles from HuggingFace until we have roughly target_tokens worth.
    Uses ccdv/pubmed-summarization (full article text) with a fallback to pubmed_qa.
    """
    from datasets import load_dataset  # imported here so the module loads without datasets installed

    print(f"  Streaming medical corpus (target: {target_tokens:,} tokens)...")

    dataset = None
    text_field = "article"

    for dataset_id, field, kwargs in [
        ("ccdv/pubmed-summarization", "article", {"split": "train", "streaming": True, "trust_remote_code": True}),
        ("scientific_papers", "article", {"name": "pubmed", "split": "train", "streaming": True, "trust_remote_code": True}),
        ("pubmed_qa", "long_answer", {"name": "pqa_labeled", "split": "train", "streaming": True, "trust_remote_code": True}),
    ]:
        try:
            dataset = load_dataset(dataset_id, **kwargs)
            text_field = field
            print(f"  Using dataset: {dataset_id}")
            break
        except Exception as exc:
            print(f"  Could not load {dataset_id}: {exc}")

    if dataset is None:
        raise RuntimeError("Could not load any medical dataset from HuggingFace.")

    articles: List[str] = []
    total_tokens = 0

    for i, item in enumerate(dataset):
        if i >= max_articles or total_tokens >= target_tokens:
            break

        text = item.get(text_field, "")
        if isinstance(text, (list, dict)):
            text = str(text)
        text = text.strip()
        if len(text) < 200:
            continue

        articles.append(text)
        total_tokens += estimate_tokens(text)

        if i % 200 == 0 and i > 0:
            print(f"    {i} articles loaded, ~{total_tokens:,} tokens")

    print(f"  Fetched {len(articles)} articles, ~{total_tokens:,} tokens total")
    return articles


def build_corpus_with_needles(
    target_tokens: int,
    cache_path: str = "results/corpus_cache.txt",
) -> Tuple[str, List[dict]]:
    """
    Build a single long medical text corpus with NEEDLES planted at specific
    percentage depths. Returns (corpus_text, needles_list).

    The finished corpus (with needles already inserted) is cached to disk so
    subsequent runs skip the HuggingFace download entirely.
    """
    import os

    if os.path.exists(cache_path):
        print(f"  Loading cached corpus from {cache_path}...")
        with open(cache_path, encoding="utf-8") as f:
            corpus = f.read()
        print(f"  Cached corpus: {len(corpus):,} chars (~{estimate_tokens(corpus):,} tokens)")
        return corpus, NEEDLES

    articles = fetch_medical_corpus(target_tokens)
    base = "\n\n---\n\n".join(articles)

    # Truncate base to target
    target_chars = target_tokens * CHARS_PER_TOKEN
    if len(base) > target_chars:
        # Trim at a paragraph boundary to avoid cutting mid-sentence
        cutoff = base.rfind("\n\n", 0, target_chars)
        base = base[: cutoff if cutoff > 0 else target_chars]

    corpus_len = len(base)
    print(f"  Base corpus: {corpus_len:,} chars (~{estimate_tokens(base):,} tokens)")

    # Insert needles at their target percentage positions within the base corpus
    needles_sorted = sorted(NEEDLES, key=lambda n: n["position_percent"])
    parts: List[str] = []
    prev = 0

    for needle in needles_sorted:
        raw_pos = int(needle["position_percent"] / 100 * corpus_len)
        # Snap forward to the next sentence end so we don't split mid-sentence
        snap = base.find(". ", raw_pos)
        insert_at = (snap + 2) if 0 < snap < corpus_len - 2 else raw_pos

        parts.append(base[prev:insert_at])
        parts.append(
            f"\n\n[CLINICAL RECORD INSERT - NEEDLE {needle['id']}]\n"
            f"{needle['text']}\n"
            f"[END CLINICAL RECORD]\n\n"
        )
        prev = insert_at

    parts.append(base[prev:])
    corpus = "".join(parts)

    final_tokens = estimate_tokens(corpus)
    print(f"  Final corpus with needles: {len(corpus):,} chars (~{final_tokens:,} tokens)")

    # Cache to disk for reproducibility and faster re-runs
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"  Corpus cached to {cache_path}")

    return corpus, NEEDLES


def truncate_to_model_limit(corpus: str, max_tokens: int) -> str:
    """Truncate corpus to fit within a model's context window."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(corpus) <= max_chars:
        return corpus
    cutoff = corpus.rfind("\n\n", 0, max_chars)
    return corpus[: cutoff if cutoff > 0 else max_chars]
