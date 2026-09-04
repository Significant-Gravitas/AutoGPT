"""Per-call token accounting, tagged by which transcript spent it.

Peak context — the largest single request any one transcript sent — is the
headline number: cumulative tokens say what a run costs, peak says whether the
architecture hits the context wall as the task grows.
"""

from dataclasses import dataclass, field

# Claude Sonnet 5, USD per million tokens.
INPUT_USD_PER_MTOK = 2.0
OUTPUT_USD_PER_MTOK = 10.0


@dataclass
class Meter:
    calls: list[dict] = field(default_factory=list)

    def record(self, transcript: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append(
            {
                "transcript": transcript,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    @property
    def model_calls(self) -> int:
        return len(self.calls)

    @property
    def input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_USD_PER_MTOK
            + self.output_tokens * OUTPUT_USD_PER_MTOK
        ) / 1_000_000

    def peak_context(self, transcript: str | None = None) -> int:
        """Largest single request, overall or for one transcript."""
        relevant = [
            c for c in self.calls if transcript is None or c["transcript"] == transcript
        ]
        return max((c["input_tokens"] for c in relevant), default=0)

    def by_transcript(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for call in self.calls:
            row = out.setdefault(
                call["transcript"], {"calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            row["calls"] += 1
            row["input_tokens"] += call["input_tokens"]
            row["output_tokens"] += call["output_tokens"]
        for name, row in out.items():
            row["peak_context"] = self.peak_context(name)
        return out
