from __future__ import annotations

from src.repositories import supabase


class EnsembleTrainingRepository:
    """Persistence for graded model-training examples."""

    def store_example(self, row: dict) -> bool:
        return supabase.insert("ensemble_training", row)
