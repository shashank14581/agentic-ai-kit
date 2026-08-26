from __future__ import annotations

import agentic_ai


def test_release_version_and_emotion_exports():
    assert agentic_ai.__version__ == "0.5.0"
    assert agentic_ai.EmotionConfig.__name__ == "EmotionConfig"
    assert agentic_ai.EmotionMemorySystem.__name__ == "EmotionMemorySystem"
    assert agentic_ai.TreeAttentionEngine.__name__ == "TreeAttentionEngine"
    assert agentic_ai.RLStateEncoder.__name__ == "RLStateEncoder"
    assert agentic_ai.LinearQLearner.__name__ == "LinearQLearner"
