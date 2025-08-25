"""N-gram model."""

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Final, List, Optional, TextIO, Tuple, Union

# (log_prob, backoff)
NgramProb = Tuple[float, Optional[float]]

NgramProbCache = Dict[Tuple[str, ...], float]

BOS: Final = "<s>"
EOS: Final = "</s>"

UNK_LOG_PROB: Final = -20


@dataclass
class Sqlite3NgramModel:
    """N-gram model stored in an sqlite3 database."""

    order: int

    words: Dict[str, str]
    """Word -> word id as str"""

    database_path: Union[str, Path]

    def get_log_prob(
        self,
        tokens: Iterable[str],
        unk_log_prob: Optional[Callable[[str], float]] = None,
        cache: Optional[NgramProbCache] = None,
    ) -> float:
        """Get log probability of token sequence relative to an n-gram model."""
        if cache is None:
            cache = {}

        conn: Optional[sqlite3.Connection] = None
        cur: Optional[sqlite3.Cursor] = None

        total_log_prob = 0.0
        context: List[str] = []

        for word in tokens:
            if word == BOS:
                # Skip BOS since its not a normal token
                context.append(word)
                continue

            context_key = tuple(context + [word])

            # Check external prefix cache
            if context_key in cache:
                total_log_prob = cache[context_key]
                context.append(word)
                continue

            found = False

            # Try highest to lowest n-gram order
            for n in reversed(range(1, self.order + 1)):
                prefix_ids = []
                prefix = tuple(context[-(n - 1) :]) if n > 1 else ()

                # Skip this n-gram order if any prefix word is unknown
                for ctx_word in prefix:
                    word_id = self.words.get(ctx_word)
                    if word_id is None:
                        break
                    prefix_ids.append(word_id)
                else:
                    prefix_id_str = " ".join(prefix_ids)
                    word_id = self.words.get(word)

                    if word_id is not None:
                        ngram_id_str = (
                            f"{prefix_id_str} {word_id}" if prefix_id_str else word_id
                        )
                        if cur is None:
                            conn, cur = self._get_cursor()

                        cur.execute(
                            "SELECT log_prob FROM ngrams WHERE word_ids = ?",
                            (ngram_id_str,),
                        )
                        row = cur.fetchone()
                        if row:
                            total_log_prob += row[0]
                            found = True
                            break  # stop backoff

                    # Backoff weight if exact ngram wasn't found
                    if cur is None:
                        conn, cur = self._get_cursor()

                    cur.execute(
                        "SELECT backoff FROM ngrams WHERE word_ids = ?",
                        (prefix_id_str,),
                    )
                    row = cur.fetchone()
                    if row and (row[0] is not None):
                        total_log_prob += row[0]

            if not found:
                if unk_log_prob is None:
                    total_log_prob += UNK_LOG_PROB
                else:
                    total_log_prob += unk_log_prob(word)

            context.append(word)

            # Store in external prefix cache
            cache[context_key] = total_log_prob

        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass  # ignore errors
            finally:
                conn = None

        return total_log_prob

    def _get_cursor(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(f"file:{self.database_path}?mode=ro", uri=True)
        return (conn, conn.cursor())


# -----------------------------------------------------------------------------


@dataclass
class MemoryNgramModel:
    """In-memory n-gram Model."""

    order: int

    # token ngram -> probability
    probs: Dict[Tuple[str, ...], NgramProb] = field(default_factory=dict)

    @staticmethod
    def from_arpa(arpa_file: TextIO) -> "MemoryNgramModel":
        """Load ngram model from ARPA file."""
        model = MemoryNgramModel(order=0)
        order = 0
        reading_ngrams = False

        for line in arpa_file:
            line = line.strip()

            # Start of new section
            if line.startswith("\\") and "-grams:" in line:
                order = int(line.strip("\\-grams:"))
                model.order = max(order, model.order)
                reading_ngrams = True
                continue

            if line.startswith("\\end\\"):
                break

            if (not line) or line.startswith("ngram") or (not reading_ngrams):
                continue

            parts = line.split()
            if len(parts) < order + 1:
                continue  # malformed line

            log_prob = float(parts[0])
            ngram = tuple(parts[1 : 1 + order])
            backoff = float(parts[1 + order]) if len(parts) > 1 + order else None

            model.probs[ngram] = (log_prob, backoff)

        return model

    def get_log_prob(
        self,
        tokens: Iterable[str],
        unk_log_prob: float = UNK_LOG_PROB,
        cache: Optional[NgramProbCache] = None,
    ) -> float:
        """Get log probability of token sequence relative to an n-gram model."""
        if cache is None:
            cache = {}

        total_log_prob = 0.0
        context: List[str] = []

        for word in tokens:
            if word == BOS:
                # Skip BOS since its not a normal token
                context.append(word)
                continue

            context_key = tuple(context + [word])

            # Check external prefix cache
            if context_key in cache:
                total_log_prob = cache[context_key]

                context.append(word)
                continue

            found = False

            # Try highest to lowest n-gram order
            for n in reversed(range(1, self.order + 1)):
                prefix = tuple(context[-(n - 1) :]) if n > 1 else ()
                ngram = prefix + (word,)

                if ngram in self.probs:
                    total_log_prob += self.probs[ngram][0]
                    found = True
                    break

                if (prefix_info := self.probs.get(prefix)) and (
                    prefix_info[1] is not None
                ):
                    # backoff
                    total_log_prob += prefix_info[1]

            if not found:
                total_log_prob += unk_log_prob

            context.append(word)

            # Store in external prefix cache
            cache[context_key] = total_log_prob

        return total_log_prob
