from __future__ import annotations

import json

from src.model.decomposition import ProbabilityDecomposition
from src.repositories import supabase


class ResearchRepository:
    """Supabase-backed write access for research predictions."""

    def store_prediction(
        self,
        game_id: int,
        decomposition: ProbabilityDecomposition,
        edge_type: str = "B",
        recommendation: str = "MONITOR",
        discord_thread_id: str | None = None,
        model_variant: str = "base",
    ) -> bool:
        active_edge = decomposition.edge
        if decomposition.bet_side == "away" and decomposition.away_edge:
            active_edge = decomposition.away_edge

        row = {
            "game_id": game_id,
            "prob_decomposition": json.dumps(decomposition.to_supabase_dict()),
            "edge_type": edge_type,
            "effective_edge": round(active_edge.effective_edge, 4) if active_edge else None,
            "recommendation": recommendation,
            "conditional_scenarios": "[]",
            "data_freshness_status": f"model_variant={model_variant}",
        }
        if discord_thread_id:
            row["discord_thread_id"] = discord_thread_id
        return supabase.insert("research", row)
