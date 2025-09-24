"""Tool wrapping a Vision Transformer (emotion classifier) to produce coarse sentiment.

This integrates with `smolagents` Tool interface so the agent can invoke it.
It performs the following steps:
1. Load (lazy) a Hugging Face image classification pipeline.
2. Accept an image reference (path or base64) plus optional user notes.
3. Run top-K emotion predictions.
4. Map fine-grained emotions to coarse sentiment (positive/negative/neutral).
5. Return a structured JSON-ready dict with probabilities and reasoning.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from smolagents import Tool
from transformers import pipeline

from config.settings import (
    VIT_EMOTION_MODEL,
    EMOTION_TO_SENTIMENT,
    SentimentResult,
)
from utils.image_utils import load_image, resize_if_needed


class SentimentViTTool(Tool):
    name = "sentiment_vit_tool"
    description = (
        "Classify the coarse facial sentiment (positive/negative/neutral) of a person in an image. "
        "Input should be either a filesystem path or a base64 encoded image string. "
        "Returns JSON with sentiment, dominant_emotion, confidence, and emotion probabilities."
    )
    inputs = {
        "image": {
            "type": "string",
            "description": "Path or base64 image string",
            "required": True,
        },
        "notes": {
            "type": "string",
            "description": "Optional context or user comment about the image",
            "required": False,
            "nullable": True,
        },
    }
    output_type = "string"  # JSON string

    def __init__(self) -> None:  # Lazy pipeline init
        super().__init__()
        self._pipe = None

    def _load_pipeline(self):
        if self._pipe is None:
            self._pipe = pipeline(
                task="image-classification",
                model=VIT_EMOTION_MODEL,
            )
        return self._pipe

    def forward(self, image: str, notes: str | None = None) -> str:
        pil_image = load_image(image)
        pil_image = resize_if_needed(pil_image)
        pipe = self._load_pipeline()
        raw_preds: List[Dict[str, Any]] = pipe(pil_image)

        # Normalize predictions into label -> score
        emotions: Dict[str, float] = {}
        for item in raw_preds:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            emotions[label] = max(score, emotions.get(label, 0.0))

        if not emotions:
            result = SentimentResult(
                sentiment="unknown",
                emotions={},
                dominant_emotion="unknown",
                confidence=0.0,
                reasoning="Model returned no predictions.",
            )
            return json.dumps(result.to_dict())

        dominant_emotion = max(emotions.items(), key=lambda kv: kv[1])[0]
        dominant_conf = emotions[dominant_emotion]

        # Aggregate: majority vote by highest cumulative probability per sentiment
        sentiment_scores: Dict[str, float] = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for label, score in emotions.items():
            coarse = EMOTION_TO_SENTIMENT.get(label, "unknown")
            sentiment_scores[coarse] = sentiment_scores.get(coarse, 0.0) + score

        final_sentiment = max(sentiment_scores.items(), key=lambda kv: kv[1])[0]
        final_confidence = sentiment_scores[final_sentiment]

        reasoning_parts = [
            f"Dominant emotion: {dominant_emotion} ({dominant_conf:.2f})",
            f"Aggregated sentiment scores: {sentiment_scores}",
        ]
        if notes:
            reasoning_parts.append(f"User notes considered: {notes[:120]}")

        # Simple uncertainty heuristic
        if dominant_conf < 0.4:
            reasoning_parts.append("Low confidence: top emotion probability < 0.40")

        result = SentimentResult(
            sentiment=final_sentiment,
            emotions=emotions,
            dominant_emotion=dominant_emotion,
            confidence=final_confidence,
            reasoning=" | ".join(reasoning_parts),
        )
        return json.dumps(result.to_dict())


__all__ = ["SentimentViTTool"]
