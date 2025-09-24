"""Central configuration for the Sentiment Vision Agent.

This module centralizes tunable constants so they can be adjusted in one place.
Future extensions (e.g. RAG embedding model, vector store params) should be
added here with clear naming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# Model identifiers (Hugging Face Hub). The emotion model returns fine-grained emotions.
# We will map them to coarse sentiment labels (positive / negative / neutral).
VIT_EMOTION_MODEL = "dima806/facial_emotions_image_detection"  # Example ViT-based emotion classifier
# Alternative candidates for experimentation (documented for future work):
# - "trpakov/vit-face-expression" (ViT fine-tuned on face expressions)
# - "posexpression/facial_emotion_recognition" (non ViT baseline for comparison)

# Sentiment mapping. Keys are lowercase emotion labels produced by the model's id2label.
# Values are one of: "positive", "negative", "neutral".
EMOTION_TO_SENTIMENT: Dict[str, str] = {
    # Positive cluster
    "happy": "positive",
    "surprised": "positive",  # surprise often neutral/positive; treat as positive here
    # Negative cluster
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    # Neutral cluster
    "neutral": "neutral",
    # Fallback emotions (if any new labels appear in alternative models)
    "contempt": "negative",
    "confused": "neutral",
}

DEFAULT_SENTIMENT_LABELS: List[str] = ["positive", "negative", "neutral"]

# System prompt for the agent. This is a crucial part of the agent's behavior.
FULL_SYSTEM_PROMPT = """
You are SentimentVision, a polite, concise, multimodal AI assistant.

You have access to the following tools: {tools}
Conversation memory: {chat_history}

Use the following loop format exactly in the same order:
    Question: the user's input
    Thought: reasoning about what to do next
    Action: the tool name, MUST be one of [{tool_names}]
    Action Input: valid input for the tool
    Observation: result of the action
    ... (repeat Thought/Action/Action Input/Observation as needed)
    Thought: I now know the final answer
    Final Answer: your final answer to the user

Begin!
"""

@dataclass(frozen=True)
class SentimentResult:
    sentiment: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "sentiment": self.sentiment,
            "emotions": self.emotions,
            "dominant_emotion": self.dominant_emotion,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


__all__ = [
    "VIT_EMOTION_MODEL",
    "EMOTION_TO_SENTIMENT",
    "DEFAULT_SENTIMENT_LABELS",
    "FULL_SYSTEM_PROMPT",
    "SentimentResult",
]