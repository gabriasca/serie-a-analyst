from __future__ import annotations

from typing import Any


def build_prediction_explanation(prediction: dict[str, Any]) -> str:
    if not prediction.get("ok"):
        return prediction.get("message", "Dati insufficienti per costruire una spiegazione.")

    factors = prediction["factors"]
    league = factors["league"]
    home = factors["home_team"]
    away = factors["away_team"]

    return (
        f"Il modello parte da una media campionato di {league['avg_goals_per_team']:.2f} gol per squadra "
        f"e da un vantaggio casa stimato in {league['home_advantage']:.2f}. "
        f"La squadra di casa mostra una forza offensiva di {home['attack_strength']:.2f} e una forma recente "
        f"di {home['form_string']} (fattore {home['form_factor']:.2f}), mentre la squadra ospite arriva con "
        f"una forza offensiva di {away['attack_strength']:.2f} e una forma di {away['form_string']} "
        f"(fattore {away['form_factor']:.2f}). "
        f"Dal lato difensivo, valori piu bassi indicano una difesa piu solida: "
        f"casa {home['defense_strength']:.2f}, trasferta {away['defense_strength']:.2f}. "
        f"Combinando questi fattori, il modello stima {prediction['expected_goals_home']:.2f} expected goals "
        f"per la squadra di casa e {prediction['expected_goals_away']:.2f} per la squadra in trasferta. "
        "Questa e una stima statistica basata sui dati disponibili, non una certezza."
    )
