from __future__ import annotations

from datetime import date, timedelta

from src.domain.prediction_record import PredictionRecord
from src.repositories import supabase


class PredictionRecordRepository:
    def list_records(self, start_date: date, end_date: date | None = None) -> list[PredictionRecord]:
        if not supabase.is_configured():
            return []
        end = end_date or (start_date + timedelta(days=1))
        rows = supabase.select(
            "prediction_records",
            {
                "select": "*",
                "created_at": f"gte.{start_date.isoformat()}T00:00:00",
                "order": "created_at.asc",
            },
        )
        parsed = []
        for row in rows:
            created_at = row.get("created_at", "")
            if end and created_at and created_at[:10] >= end.isoformat():
                continue
            record = PredictionRecord.from_supabase_row(row)
            if record:
                parsed.append(record)
        return parsed
