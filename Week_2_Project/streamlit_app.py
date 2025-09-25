"""Streamlit app for CheerSearch agent.

Features:
- Chat UI for user messages.
- Image upload widget; uploads saved to `data/uploads`.
- When an image is uploaded, the app calls the sentiment tool and shows the JSON result.
- If `GROQ_API_KEY` is set, uses the full agent via `agents.sentiment_agent.build_agent` to allow multi-tool conversations.
- Otherwise, calls `tools.sentiment_vit_tool.SentimentViTTool` directly.

Run: `streamlit run streamlit_app.py`
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import streamlit as st

from agents.cheersearch_agent import build_agent, sentiment_tool
from tools.sentiment_vit_tool import SentimentViTTool

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
WORKSPACE_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = WORKSPACE_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="CheerSearch", layout="wide")
st.title("CheerSearch — Happily Productive ☺️")


# ---------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------
def extract_reply(result):
    """Normalize reply output from agent/tool into a string."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        candidate = result.get("output")
        if candidate is not None:
            return extract_reply(candidate)
        # fallback: first string-like value
        for v in result.values():
            if isinstance(v, str):
                return v
            if isinstance(v, dict) and "text" in v and isinstance(v["text"], str):
                return v["text"]
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    return str(result)


def parse_result(result):
    """Parse tool result into JSON/dict if possible."""
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return result
    return result


def format_sentiment_result(parsed_result):
    """Format sentiment analysis result in a user-friendly way."""
    if isinstance(parsed_result, dict):
        # Check for error first
        if "error" in parsed_result:
            return f"❌ **Error:** {parsed_result.get('message', 'Unknown error')}"

        # Format normal sentiment result
        sentiment = parsed_result.get("sentiment", "unknown")
        confidence = parsed_result.get("confidence", 0)
        dominant_emotion = parsed_result.get("dominant_emotion", "unknown")
        emotions = parsed_result.get("emotions", {})

        # Emoji mapping for sentiment
        sentiment_emoji = {
            "positive": "😊",
            "negative": "😔",
            "neutral": "😐"
        }

        # Format the result
        emoji = sentiment_emoji.get(sentiment, "❓")
        result_text = f"{emoji} **Overall Sentiment:** {sentiment.title()}\n\n"
        result_text += f"**Confidence:** {confidence:.1%}\n\n"
        result_text += f"**Dominant Emotion:** {dominant_emotion.title()}\n\n"

        if emotions:
            result_text += "**Emotion Breakdown:**\n"
            # Sort emotions by confidence
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            for emotion, score in sorted_emotions[:5]:  # Show top 5
                bar_length = int(score * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                result_text += f"- {emotion.title()}: {bar} {score:.1%}\n"

            # If the most dominant emotion is not positive, add an empathetic note
            if sentiment == "negative":
                result_text += "\nI'm sorry to see that you're feeling this way. Remember, it's okay to have tough days. You're doing great, and things will get better! 🌟"

        return result_text
    else:
        return str(parsed_result)





# ---------------------------------------------------------------------
# Sidebar — Settings
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    groq_key = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")
    model_options = [
        "gemma2-9b-it",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "moonshotai/kimi-k2-instruct",
        "moonshotai/kimi-k2-instruct-0905",
    ]
    default_index = (
        model_options.index("gemma2-9b-it")
        if "gemma2-9b-it" in model_options
        else 0
    )
    model_name = st.selectbox("Model", options=model_options, index=default_index)
    max_steps = st.number_input("Agent max steps", min_value=1, max_value=20, value=4)

    if groq_key:
        st.success("Agent API key loaded from environment.")
    else:
        st.info("No GROQ_API_KEY found in environment; running local sentiment tool only.")


# ---------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of (role, text)

if (
    "agent" not in st.session_state
    or st.session_state.get("_agent_model_name") != model_name
):
    if groq_key:
        try:
            agent_exec = build_agent(
                groq_api_key=groq_key, model_name=model_name, max_steps=max_steps
            )
            st.session_state["agent"] = agent_exec
            st.session_state["use_agent"] = True
            st.session_state["_agent_model_name"] = model_name
        except Exception as e:
            st.warning(f"Could not initialize agent: {e}. Falling back to local tool.")
            st.session_state["agent"] = None
            st.session_state["use_agent"] = False
            st.session_state["_agent_model_name"] = None
    else:
        st.session_state["agent"] = None
        st.session_state["use_agent"] = False
        st.session_state["_agent_model_name"] = None


# ---------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------
col1, col2 = st.columns([3, 1])

# --- Chat ---
with col1:
    # Header with buttons
    col_title, col_buttons = st.columns([2, 1])
    with col_title:
        st.subheader("Conversation")
    with col_buttons:
        # Use columns for side-by-side buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🗑️ Clear", help="Clear conversation history"):
                st.session_state["chat_history"] = []
                st.session_state["processing"] = False
                st.session_state["pending_input"] = None
                st.rerun()
        with btn_col2:
            if st.button("🧠 Forget", help="Reset agent memory/context"):
                # Clear agent memory if available
                if st.session_state.get("agent") and hasattr(st.session_state["agent"], "memory"):
                    try:
                        st.session_state["agent"].memory.clear()
                    except Exception:
                        pass
                # Force agent rebuild to reset context
                st.session_state.pop("agent", None)
                st.session_state.pop("_agent_model_name", None)
                st.session_state["processing"] = False
                st.session_state["pending_input"] = None
                st.rerun()

    # Initialize processing state
    if "processing" not in st.session_state:
        st.session_state["processing"] = False
    if "pending_input" not in st.session_state:
        st.session_state["pending_input"] = None

    # Create a scrollable chat area with fixed height
    chat_container = st.container(height=400)
    with chat_container:
        if st.session_state["chat_history"]:
            for role, text in st.session_state["chat_history"]:
                if role == "user":
                    st.chat_message("user").write(text)
                else:
                    st.chat_message("assistant").write(text)
        else:
            st.info("Start a conversation by typing a message below...")
        
        # Show loading spinner if processing
        if st.session_state["processing"]:
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    # Process the pending input
                    if st.session_state["pending_input"]:
                        input_text = st.session_state["pending_input"]
                        
                        # Generate response
                        if st.session_state.get("use_agent") and st.session_state.get("agent"):
                            try:
                                agent = st.session_state["agent"]
                                result = agent.invoke({"input": input_text})
                                reply = extract_reply(result)
                                if isinstance(reply, dict) and "text" in reply:
                                    reply = reply["text"]
                                st.session_state["chat_history"].append(("assistant", reply))
                            except Exception as e:
                                st.session_state["chat_history"].append(("assistant", f"Agent error: {e}"))
                        else:
                            st.session_state["chat_history"].append(("assistant", "(Local mode) Received your message."))
                        
                        # Reset processing state
                        st.session_state["processing"] = False
                        st.session_state["pending_input"] = None
                        st.rerun()

    # Input area at the bottom (Streamlit's chat_input)
    user_input = st.chat_input("Type your message here...", disabled=st.session_state["processing"])

    # Process message when submitted
    if user_input and not st.session_state["processing"]:
        # Add user message to history immediately
        st.session_state["chat_history"].append(("user", user_input))
        
        # Set up processing state
        st.session_state["processing"] = True
        st.session_state["pending_input"] = user_input
        
        # Force rerun to show the user message and start processing
        st.rerun()


# --- Image Upload ---
with col2:
    st.subheader("Stay Cheerful!")
    st.write("Upload an image of yourself. I need to make sure you're in a good mood while researching! 😊")
    uploaded = st.file_uploader(
        "Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"]
    )
    # Notes input that updates automatically (no Enter required)
    notes = st.text_input(
        "Notes (optional)",
        key="image_notes",
        placeholder="Add context about the image...",
        help="Optional notes to provide context for the sentiment analysis"
    )

    if uploaded is not None:
        # Save upload safely
        safe_name = uploaded.name.replace("..", "_").replace("/", "_").replace("\\", "_")
        save_path = UPLOAD_DIR / safe_name
        counter = 1
        while save_path.exists():
            save_path = UPLOAD_DIR / f"{save_path.stem}_{counter}{save_path.suffix}"
            counter += 1
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())

        st.image(str(save_path), caption="Uploaded image", use_container_width=True)
        st.success(f"Image saved successfully!")

        # Store image path in session state
        st.session_state["current_image_path"] = str(save_path)

        # Button to analyze sentiment
        if st.button("🔍 Analyze Sentiment"):
            with st.spinner("Analyzing sentiment..."):
                try:
                    if st.session_state.get("use_agent") and st.session_state.get("agent"):
                        func = getattr(sentiment_tool, "func", None)
                        if callable(func):
                            result = func(image=str(save_path), notes=notes)
                        else:
                            result = SentimentViTTool().forward(str(save_path), notes=notes)
                    else:
                        result = SentimentViTTool().forward(str(save_path), notes=notes)

                    parsed = parse_result(result)

                    # Display formatted result
                    st.markdown("### 📊 Sentiment Analysis Results")
                    formatted_result = format_sentiment_result(parsed)
                    st.markdown(formatted_result)

                    # Show raw JSON in expander for technical users
                    with st.expander("📋 Raw Analysis Data"):
                        st.json(parsed)

                except Exception as e:
                    st.error(f"❌ Error during sentiment analysis: {e}")


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("CheerSearch — Streamlit UI")
