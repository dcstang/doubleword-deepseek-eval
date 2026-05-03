"""
Builds a long-context corpus structured as a ward round's clinical note collection.

Source: AGBonnet/augmented-clinical-notes (HuggingFace)
Structure: numbered clinical notes separated by note headers, with 5 needle notes
           planted at specific depths. The needles look identical to real notes so
           the model must read carefully to find the planted facts.

Cache: the finished corpus is saved to results/corpus_cache.txt on first build
       and reloaded on subsequent runs (delete the file to force a fresh download).
"""

import os
import random
from typing import List, Tuple

CHARS_PER_TOKEN = 4  # rough estimate for English clinical text

CONTEXT_TIERS = [200_000, 400_000, 600_000, 800_000]
# Different seed per tier so each tier gets a different random sample of notes,
# but both models are always built from the same seed at each tier.
TIER_SEEDS = {200_000: 42, 400_000: 137, 600_000: 256, 800_000: 512}

# Needle notes look like real ward notes. Questions target the specific numeric
# or textual fact buried in each one.
NEEDLES = [
    {
        "id": 1,
        "position_percent": 5,
        "note_text": (
            "ICU ADMISSION NOTE\n"
            "Patient: John Anderson  |  MRN: 78234  |  DOB: 15-Apr-1957 (67M)\n"
            "Attending: Dr R. Patel  |  Unit: Medical ICU, Bed 4B\n"
            "Date/Time: 14-Mar-2025  07:32\n\n"
            "ASSESSMENT:\n"
            "Mr Anderson is a 67-year-old male admitted from ED with septic shock "
            "secondary to gram-negative bacteraemia (blood cultures pending). On arrival "
            "to the ICU his Sequential Organ Failure Assessment (SOFA) score was 12, "
            "reflecting multi-organ dysfunction. He was commenced on norepinephrine at "
            "0.15 mcg/kg/min for vasopressor support and empirical broad-spectrum "
            "antibiotics (piperacillin-tazobactam plus vancomycin).\n\n"
            "PLAN: Serial lactates q4h, repeat cultures at 48h, daily SOFA reassessment."
        ),
        "question": "What was the SOFA score documented for patient John Anderson (MRN-78234) on his ICU admission note dated 14-Mar-2025?",
        "expected_answer": "12",
    },
    {
        "id": 2,
        "position_percent": 25,
        "note_text": (
            "CARDIOLOGY MDT MEETING NOTE\n"
            "Date: 18-Mar-2025  14:00  |  Chair: Dr L. Okonkwo\n"
            "Attendees: Cardiology, Haematology, Clinical Pharmacology\n\n"
            "AGENDA ITEM 3 — CARDIO-447 TRIAL RESULTS (circulated pre-meeting):\n"
            "The CARDIO-447 RCT evaluated novel anticoagulation in high-risk AF. "
            "Treatment arm B (factor Xa inhibitor plus antiplatelet therapy) achieved "
            "a 90-day all-cause mortality rate of 23.4%, versus 31.2% in the control "
            "arm on standard anticoagulation alone (p=0.003, NNT=13). "
            "Enrolment: 2,847 patients across 47 centres.\n\n"
            "ACTION: Pharmacy to draft formulary amendment proposal by 01-Apr-2025."
        ),
        "question": "According to the Cardiology MDT meeting note dated 18-Mar-2025, what was the 90-day mortality rate in treatment arm B of the CARDIO-447 trial?",
        "expected_answer": "23.4%",
    },
    {
        "id": 3,
        "position_percent": 50,
        "note_text": (
            "QUALITY IMPROVEMENT MEETING — ACUTE MEDICINE\n"
            "Date: 21-Mar-2025  09:00  |  Facilitator: Dr A. Williamson\n\n"
            "AGENDA ITEM 2 — BENCHMARKING LENGTH OF STAY:\n"
            "Presented benchmarking data from the Oslo Cardiac Registry (2019–2023), "
            "encompassing 4,847 consecutive admissions for acute coronary syndromes. "
            "Reported median hospital length of stay: 14.2 days "
            "(interquartile range 8 to 21 days). Independent predictors of prolonged "
            "stay included diabetes mellitus (OR 1.87), chronic renal dysfunction "
            "(OR 2.34), and in-hospital procedural complications (OR 3.12).\n\n"
            "ACTION: Compare our unit's current median LOS against Oslo benchmark."
        ),
        "question": "What median hospital length of stay and IQR were reported from the Oslo Cardiac Registry, as presented in the Quality Improvement meeting note dated 21-Mar-2025?",
        "expected_answer": "14.2 days, IQR 8-21 days",
    },
    {
        "id": 4,
        "position_percent": 75,
        "note_text": (
            "ICU PROTOCOL IMPLEMENTATION NOTE\n"
            "Date: 23-Mar-2025  |  Author: ICU Lead Nurse, countersigned Dr S. Chen\n\n"
            "RE: SEPSIS RESUSCITATION BUNDLE — VASOPRESSOR TITRATION GUIDELINE\n"
            "As per Dr Sarah Chen's validated sepsis resuscitation protocol (published "
            "Critical Care Medicine, 2024), all vasopressor titration decisions on this "
            "unit must target a mean arterial pressure (MAP) of greater than 65 mmHg. "
            "Any deviation below this threshold sustained for more than 15 consecutive "
            "minutes must trigger immediate reassessment and escalation to second-line "
            "vasopressors per the attached algorithm.\n\n"
            "All nursing staff to complete acknowledgement sign-off by 30-Mar-2025."
        ),
        "question": "According to the ICU protocol note dated 23-Mar-2025 referencing Dr Sarah Chen's sepsis protocol, what MAP threshold must vasopressor titration target?",
        "expected_answer": "MAP > 65 mmHg",
    },
    {
        "id": 5,
        "position_percent": 95,
        "note_text": (
            "RESPIRATORY MEDICINE GRAND ROUND NOTE\n"
            "Date: 28-Mar-2025  12:00  |  Presenter: Dr F. Osei-Bonsu\n\n"
            "TOPIC: COMBINATION THERAPY IN SEVERE CAP — RESPIRE-2024 TRIAL\n"
            "Dr Osei-Bonsu presented the RESPIRE-2024 multicentre RCT (n=892) evaluating "
            "combination antimicrobial therapy in severe community-acquired pneumonia. "
            "The AZITHRO-DEXA-IV protocol (azithromycin 500 mg plus dexamethasone 10 mg "
            "IV daily) achieved a clinical response rate of 87.3% at day 7, significantly "
            "outperforming standard beta-lactam monotherapy at 68.1% (p<0.001, OR 3.21).\n\n"
            "DISCUSSION: Protocol adoption to be reviewed at next formulary meeting."
        ),
        "question": "What day-7 clinical response rate did the AZITHRO-DEXA-IV protocol achieve in the RESPIRE-2024 trial, as presented in the Grand Round note dated 28-Mar-2025?",
        "expected_answer": "87.3%",
    },
]


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _format_note(index: int, text: str) -> str:
    return f"=== NOTE {index} ===\n{text.strip()}\n=== END NOTE ===\n"


def fetch_clinical_notes(
    target_tokens: int,
    seed: int = 0,
    max_notes: int = 50_000,
) -> List[str]:
    """
    Stream clinical notes from AGBonnet/augmented-clinical-notes until we have
    roughly target_tokens worth of text.

    seed: when non-zero, shuffles the streaming dataset with a fixed seed so
          different tiers pull different note samples while remaining reproducible.
    """
    from datasets import load_dataset

    print(f"  Streaming clinical notes (target: {target_tokens:,} tokens, seed={seed})...")

    dataset = None
    text_field = None

    candidates = [
        (
            "AGBonnet/augmented-clinical-notes",
            None,
            {"split": "train", "streaming": True, "trust_remote_code": True},
        ),
        # fallbacks if the primary dataset is unavailable
        (
            "AGBonnet/augmented-clinical-notes",
            None,
            {"split": "test", "streaming": True, "trust_remote_code": True},
        ),
        (
            "medalpaca/medical_meadow_medical_flashcards",
            "output",
            {"split": "train", "streaming": True, "trust_remote_code": True},
        ),
    ]

    for dataset_id, field_hint, kwargs in candidates:
        try:
            ds = load_dataset(dataset_id, **kwargs)
            if seed:
                ds = ds.shuffle(seed=seed, buffer_size=10_000)
            # Probe the first item to find the text field
            first = next(iter(ds))
            if field_hint and field_hint in first:
                text_field = field_hint
            else:
                # Pick the longest string field as the note body
                text_field = max(
                    (k for k, v in first.items() if isinstance(v, str)),
                    key=lambda k: len(first[k]),
                    default=None,
                )
            if text_field is None:
                raise ValueError("No string field found in dataset")
            dataset = ds
            print(f"  Using dataset: {dataset_id}  (field: '{text_field}')")
            break
        except Exception as exc:
            print(f"  Could not load {dataset_id}: {exc}")

    if dataset is None:
        raise RuntimeError("Could not load any clinical note dataset from HuggingFace.")

    notes: List[str] = []
    total_tokens = 0

    for i, item in enumerate(dataset):
        if i >= max_notes or total_tokens >= target_tokens:
            break

        text = item.get(text_field, "")
        if isinstance(text, (list, dict)):
            text = str(text)
        text = text.strip()
        if len(text) < 100:
            continue

        notes.append(text)
        total_tokens += estimate_tokens(text)

        if i % 500 == 0 and i > 0:
            print(f"    {i} notes loaded, ~{total_tokens:,} tokens")

    print(f"  Fetched {len(notes)} notes, ~{total_tokens:,} tokens total")
    return notes


def build_corpus_with_needles(
    target_tokens: int,
    seed: int = 0,
    cache_path: str = None,
) -> Tuple[str, List[dict]]:
    """
    Build a corpus of numbered clinical notes with 5 needle notes planted at
    specific depth positions. Returns (corpus_text, needles_list).

    Structure:
        Ward Round Documentation header
        === NOTE 1 === ... === END NOTE ===
        === NOTE 2 === ... === END NOTE ===
        ...   ← needle notes embedded here, indistinguishable from real ones
        ...
        === NOTE N === ... === END NOTE ===

    The finished corpus is cached to cache_path for reproducible re-runs.
    Delete the cache file to force a fresh download.

    cache_path defaults to results/corpus_cache_{target_tokens//1000}k.txt so each
    tier has its own cache file and won't be overwritten by other tiers.
    """
    if cache_path is None:
        label = f"{target_tokens // 1000}k"
        cache_path = f"results/corpus_cache_{label}.txt"

    if os.path.exists(cache_path):
        print(f"  Loading cached corpus from {cache_path}...")
        with open(cache_path, encoding="utf-8") as f:
            corpus = f.read()
        print(f"  Cached corpus: {len(corpus):,} chars (~{estimate_tokens(corpus):,} tokens)")
        return corpus, NEEDLES

    notes = fetch_clinical_notes(target_tokens, seed=seed)
    n_notes = len(notes)

    if n_notes == 0:
        raise RuntimeError("No clinical notes fetched — cannot build corpus.")

    # Work out which note indices will be replaced by needle notes
    needle_indices: dict[int, dict] = {}
    for needle in NEEDLES:
        idx = int(needle["position_percent"] / 100 * n_notes)
        idx = max(0, min(idx, n_notes - 1))
        # Avoid collisions
        while idx in needle_indices:
            idx += 1
        needle_indices[idx] = needle

    # Build the corpus note-by-note
    header = (
        "WARD ROUND DOCUMENTATION\n"
        f"Total notes in this session: {n_notes + len(NEEDLES)}\n"
        "=" * 60 + "\n\n"
    )
    parts = [header]
    note_counter = 1

    for i, note_text in enumerate(notes):
        # Insert needle note before the real note at this index
        if i in needle_indices:
            needle = needle_indices[i]
            parts.append(_format_note(note_counter, needle["note_text"]))
            note_counter += 1

        parts.append(_format_note(note_counter, note_text))
        note_counter += 1

    # Any needles mapped beyond the last note go at the end
    for idx in sorted(needle_indices):
        if idx >= n_notes:
            parts.append(_format_note(note_counter, needle_indices[idx]["note_text"]))
            note_counter += 1

    corpus = "".join(parts)
    final_tokens = estimate_tokens(corpus)
    print(f"  Final corpus: {note_counter - 1} notes, {len(corpus):,} chars (~{final_tokens:,} tokens)")

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(corpus)
    print(f"  Corpus cached to {cache_path}")

    return corpus, NEEDLES


def truncate_to_model_limit(corpus: str, max_tokens: int) -> str:
    """Truncate corpus to fit within a model's context window, cutting at a note boundary."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(corpus) <= max_chars:
        return corpus
    # Prefer cutting at an END NOTE boundary
    cutoff = corpus.rfind("=== END NOTE ===", 0, max_chars)
    if cutoff > 0:
        return corpus[: cutoff + len("=== END NOTE ===")]
    return corpus[:max_chars]
