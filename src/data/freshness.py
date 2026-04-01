"""
Freshness checker: cross-validate sources and flag disagreements.

Rule from gameplan: if ESPN says player is active and balldontlie hasn't
updated in 3+ hours, flag it — disagreement is the signal, not scrape frequency.
"""

from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field


@dataclass
class FreshnessRecord:
    source: str
    data_type: str
    fetched_at: datetime
    key: str  # identifier (player name, team, game id, etc.)
    value: str  # the data point being tracked (e.g. "active", "out", "GTD")


@dataclass
class Disagreement:
    key: str
    data_type: str
    sources: list[FreshnessRecord]
    detected_at: datetime
    description: str


class FreshnessChecker:
    def __init__(self, stale_threshold_hours: float = 3.0):
        self.stale_threshold = timedelta(hours=stale_threshold_hours)
        self.records: dict[str, list[FreshnessRecord]] = {}
        self.disagreements: list[Disagreement] = []

    def record(self, source: str, data_type: str, key: str, value: str,
               fetched_at: datetime | None = None) -> None:
        """Record a data point from a source."""
        rec = FreshnessRecord(
            source=source,
            data_type=data_type,
            fetched_at=fetched_at or datetime.now(timezone.utc),
            key=key,
            value=value.strip().lower(),
        )
        compound_key = f"{data_type}:{key}"
        if compound_key not in self.records:
            self.records[compound_key] = []
        # Replace existing record from same source
        self.records[compound_key] = [
            r for r in self.records[compound_key] if r.source != source
        ]
        self.records[compound_key].append(rec)

    def check_all(self) -> list[Disagreement]:
        """Check all tracked data points for disagreements and staleness."""
        now = datetime.now(timezone.utc)
        self.disagreements = []

        for compound_key, records in self.records.items():
            if len(records) < 2:
                continue

            # Check for value disagreements
            values = {r.value for r in records}
            if len(values) > 1:
                self.disagreements.append(Disagreement(
                    key=compound_key,
                    data_type=records[0].data_type,
                    sources=list(records),
                    detected_at=now,
                    description=self._describe_disagreement(records),
                ))

            # Check for staleness (one source updated, another hasn't in 3+ hours)
            sorted_recs = sorted(records, key=lambda r: r.fetched_at, reverse=True)
            newest = sorted_recs[0]
            for older in sorted_recs[1:]:
                age = newest.fetched_at - older.fetched_at
                if age > self.stale_threshold:
                    self.disagreements.append(Disagreement(
                        key=compound_key,
                        data_type=records[0].data_type,
                        sources=[newest, older],
                        detected_at=now,
                        description=(
                            f"Stale data: {older.source} last updated "
                            f"{age.total_seconds() / 3600:.1f}h ago, "
                            f"{newest.source} updated {(now - newest.fetched_at).total_seconds() / 60:.0f}m ago"
                        ),
                    ))

        return self.disagreements

    def _describe_disagreement(self, records: list[FreshnessRecord]) -> str:
        parts = [f"{r.source}={r.value}" for r in records]
        return f"Value mismatch on {records[0].key}: {', '.join(parts)}"

    def get_timestamps(self) -> dict[str, dict[str, str]]:
        """Get freshness timestamps per source for reporting."""
        result = {}
        for compound_key, records in self.records.items():
            result[compound_key] = {}
            for r in records:
                result[compound_key][r.source] = r.fetched_at.isoformat()
        return result

    def clear(self) -> None:
        self.records.clear()
        self.disagreements.clear()
