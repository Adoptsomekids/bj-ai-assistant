"""
bj_assistant.ai
---------------
AI/ML layer for the BJ assistant.

Three components:
  1. mc_trainer.py  — offline Monte Carlo simulator that builds a Q-table
  2. kelly.py       — Kelly Criterion + Hi-Lo TC bet sizing
  3. ai_advisor.py  — runtime decision layer (Q-table lookup → fallback Basic Strategy)

Usage (training, run once):
    python -m bj_assistant.ai.mc_trainer --episodes 10000 --out models/bj_qtable.npy

Usage (runtime, already integrated in engine.py):
    from bj_assistant.ai.ai_advisor import AIAdvisor
    advisor = AIAdvisor.load("models/bj_qtable.npy")
    action = advisor.decide(player_total, is_soft, dealer_upcard, true_count, can_double, can_split)
"""
