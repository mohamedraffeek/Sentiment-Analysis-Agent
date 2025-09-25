"""Central configuration for the Sentiment Vision Agent.

This module centralizes tunable constants so they can be adjusted in one place.
Future extensions (e.g. RAG embedding model, vector store params) should be
added here with clear naming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

VIT_EMOTION_MODEL = "dima806/facial_emotions_image_detection"  # Example ViT-based emotion classifier

# Sentiment mapping. Keys are lowercase emotion labels produced by the model's id2label.
# Values are one of: "positive", "negative", "neutral".
EMOTION_TO_SENTIMENT: Dict[str, str] = {
    # Positive
    "happy": "positive",
    "surprised": "positive",
    # Negative
    "angry": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sad": "negative",
    # Neutral
    "neutral": "neutral",
    # Fallback emotions (if any new labels appear)
    "contempt": "negative",
    "confused": "neutral",
}


# System prompt for the agent. This is a crucial part of the agent's behavior.
FULL_SYSTEM_PROMPT = """
You are CheerSearch, a polite, concise, multimodal AI assistant. You have the capability to help researchers find relevant \
documents and answer questions using them based on their research requirements. You can also analyze images to determine \
the coarse sentiment (positive/negative/neutral) of these researchers by examining their facial expressions from images \
they provide. You want to help them be productive and happy while doing research. If you detect negative sentiment, you should \
offer empathetic and encouraging responses to help improve their mood.

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
    "FULL_SYSTEM_PROMPT",
    "SentimentResult",
]