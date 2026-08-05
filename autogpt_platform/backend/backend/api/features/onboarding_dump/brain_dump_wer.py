"""Word error rate for the brain-dump transcription eval.

An inline Levenshtein over words rather than ``jiwer``: the algorithm is
~20 lines and keeping it here means the harness runs anywhere without an
extra dependency. Used by ``brain_dump_eval.py``.
"""

import unicodedata

from pydantic import BaseModel, computed_field

# Word separators inside a token: a reference writing "well known" and a
# transcript writing "well-known" is a formatting difference, not an error.
SPACING_PUNCTUATION = frozenset({"-", "–", "—", "/", "_"})


class WordErrors(BaseModel):
    substitutions: int
    insertions: int
    deletions: int
    reference_words: int

    @computed_field
    @property
    def wer(self) -> float:
        if self.reference_words == 0:
            return 0.0
        total = self.substitutions + self.insertions + self.deletions
        return total / self.reference_words


def compute_word_errors(reference: str, hypothesis: str) -> WordErrors:
    """Count substitutions/insertions/deletions of ``hypothesis`` vs ``reference``."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    return _count_operations(ref, hyp, _edit_matrix(ref, hyp))


def normalize_words(text: str) -> list[str]:
    """Lowercase, drop punctuation and collapse whitespace into words."""
    cleaned = [
        (
            " "
            if char in SPACING_PUNCTUATION
            else "" if unicodedata.category(char).startswith("P") else char
        )
        for char in text.lower()
    ]
    return "".join(cleaned).split()


def _edit_matrix(ref: list[str], hyp: list[str]) -> list[list[int]]:
    costs = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        costs[i][0] = i
    for j in range(len(hyp) + 1):
        costs[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                continue
            costs[i][j] = 1 + min(costs[i - 1][j - 1], costs[i][j - 1], costs[i - 1][j])
    return costs


def _count_operations(
    ref: list[str], hyp: list[str], costs: list[list[int]]
) -> WordErrors:
    substitutions = insertions = deletions = 0
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and costs[i][j] == costs[i - 1][j - 1] + 1:
            substitutions += 1
            i, j = i - 1, j - 1
        elif j > 0 and costs[i][j] == costs[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    return WordErrors(
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_words=len(ref),
    )
