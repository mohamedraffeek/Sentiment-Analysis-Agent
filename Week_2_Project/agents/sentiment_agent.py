"""Construction of the SentimentVision conversational agent using LangChain + Groq.

Uses an agent executor with ReACT pattern, integrating:
- Groq LLM
- SentimentViTTool for facial sentiment analysis
- DuckDuckGo search tool for current/factual queries
"""
from __future__ import annotations

import os

import json
import re
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from agents.agent_memory import ChatHistoryMemory

from config.settings import FULL_SYSTEM_PROMPT
from tools.sentiment_vit_tool import SentimentViTTool
from tools.final_answer_tool import final_answer_tool


class SentimentToolSchema(BaseModel):
    image: str = Field(..., description="Path or base64 image string")
    notes: str | None = Field(default=None, description="Optional context or user comment about the image")

# keep single global instance to avoid reloading HF pipeline
_SENTIMENT_TOOL_INSTANCE: SentimentViTTool | None = None

def _normalize_path_from_maybe_json(image_arg: str, notes_arg: str | None = None):
    """
    Accepts:
      - plain path string
      - JSON string
      - JSON-like with single backslashes
      - model may also return the full arg object as text; attempt to extract
    Returns normalized (image, notes)
    """
    image = image_arg
    notes = notes_arg

    if isinstance(image, str) and image.strip().startswith("{"):
        try:
            data = json.loads(image)
            image = data.get("image", image)
            notes = data.get("notes", notes)
        except json.JSONDecodeError:
            # tolerant regex extraction (handles unescaped backslashes)
            img_match = re.search(r'"image"\s*:\s*"([^"\n]+)"', image)
            if img_match:
                image = img_match.group(1)
            notes_match = re.search(r'"notes"\s*:\s*"([^"\n]+)"', image)
            if notes_match:
                notes = notes_match.group(1)
        # Convert literal 'None' -> None
        if isinstance(notes, str) and notes.lower() == "none":
            notes = None

    # Trim surrounding quotes and whitespace
    if isinstance(image, str):
        image = image.strip().strip('"').strip()
    if isinstance(notes, str):
        notes = notes.strip().strip('"').strip()

    # If the path contains doubled backslashes (valid JSON), leave as-is.
    # If it contains single backslashes (raw form), leave as-is too.
    # But we can normalize doubled backslashes to single for filesystem ops:
    if isinstance(image, str) and "\\\\" in image:
        image = image.replace("\\\\", "\\")

    # If the path does not exist on disk, attempt to recover from a mangled
    # Windows path where backslashes were removed (e.g. 'C:Usersmepathfile.jpg').
    # Strategies:
    # 1. If the string contains the drive letter pattern (e.g. 'C:'), try to
    #    insert backslashes before known folder names or before capital letters.
    # 2. Search the repository (utils/ and workspace root) for filenames matching
    #    the basename and use the first match.
    try:
        if isinstance(image, str) and not image.startswith("data:") and not os.path.exists(image):
            # Quick heuristic: if it looks like a Windows path missing backslashes
            if re.match(r'^[A-Za-z]:[\\/]?.*', image) and "\\" not in image and "/" not in image:
                # Try to insert backslashes after the drive letter and before known folder names
                # Common folder names in this project: Users, Documents, Training, Week_2_Project, utils
                repaired = image
                for part in ["Users", "Documents", "Training", "Week_2_Project", "utils", "Downloads"]:
                    repaired = repaired.replace(part, "\\" + part)
                # add backslash after drive letter if missing
                repaired = re.sub(r'^([A-Za-z]:)(?!\\)', r'\\\\', repaired)
                # normalize doubled backslashes introduced above
                repaired = repaired.replace('\\\\', '\\')
                if os.path.exists(repaired):
                    image = repaired
                else:
                    # fallback: search workspace for matching filename
                    basename = os.path.basename(image)
                    if basename:
                        # search under workspace root relative to this file
                        workspace_root = os.path.dirname(os.path.dirname(__file__))
                        matches = []
                        for dirpath, _, filenames in os.walk(workspace_root):
                            if basename in filenames:
                                matches.append(os.path.join(dirpath, basename))
                        if matches:
                            image = matches[0]
    except Exception:
        # don't fail normalization on unexpected errors
        pass
    return image, notes

def _sentiment_call(image: str, notes: str | None = None) -> str:
    global _SENTIMENT_TOOL_INSTANCE
    if _SENTIMENT_TOOL_INSTANCE is None:
        _SENTIMENT_TOOL_INSTANCE = SentimentViTTool()

    image, notes = _normalize_path_from_maybe_json(image, notes)

    # optional: sanity check if it looks like a path or base64
    if isinstance(image, str) and not image.startswith("data:") and os.path.sep in image:
        if not os.path.exists(image):
            # return JSON error structure expected by downstream consumer
            error_payload = {
                "error": "image_not_found",
                "message": f"Image path not found: {image}",
                "suggestion": "Ask the user to upload or provide a valid image path/base64 before calling the tool.",
            }
            return json.dumps(error_payload)

    print(f"[Tool] sentiment_vit_tool called with image='{(image or '')[:50]}...' notes='{(notes or '')[:50]}...'")
    return _SENTIMENT_TOOL_INSTANCE.forward(image=image, notes=notes)

sentiment_tool = StructuredTool.from_function(
    name="sentiment_vit_tool",
    description=(
        "Classify facial sentiment (positive/negative/neutral) from an image path or base64 string. "
        "Returns JSON with sentiment, dominant_emotion, confidence, emotions map, reasoning."
    ),
    func=_sentiment_call,
    args_schema=SentimentToolSchema,
)

search_tool = DuckDuckGoSearchRun()

prompt = ChatPromptTemplate.from_messages([
    ("system", FULL_SYSTEM_PROMPT),
    ("human", "{input}\n{agent_scratchpad}"),
])


def build_agent(
    groq_api_key: str | None = None,
    model_name: str | None = None,
    max_steps: int | None = None,  # Kept for interface compatibility (not strictly used)
) -> AgentExecutor:
    """Build a LangChain agent executor.

    Environment precedence (if arguments not provided):
      - Key: GROQ_API_KEY then API_KEY
      - Model: GROQ_MODEL then MODEL
    """
    if groq_api_key is None:
        groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
    if not groq_api_key:
        raise ValueError("Missing Groq API key. Set GROQ_API_KEY.")

    if model_name is None:
        model_name = os.getenv("GROQ_MODEL") or os.getenv("MODEL") or "llama-3.1-8b-instant"

    llm = ChatGroq(
        model=model_name,
        temperature=0,
    )

    tools = [final_answer_tool, sentiment_tool, search_tool]

    agent = create_react_agent(llm, tools, prompt)

    memory = ChatHistoryMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=max_steps,
        handle_parsing_errors=True,
        memory=memory,
    )

    return agent_executor


__all__ = ["build_agent", "sentiment_tool"]
