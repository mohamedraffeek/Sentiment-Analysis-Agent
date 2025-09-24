# SentimentVision Agent

A modular multimodal agent that can hold conversations and classify the coarse facial sentiment (positive / negative / neutral) from an input image using a Vision Transformer (emotion classifier) wrapped as a tool via `smolagents`.

## Features
- Conversational agent (LLM) with a structured system prompt.
- Vision Transformer emotion model mapped to sentiment.
- Image input: file path or base64 (data URI or raw string).
- Structured JSON output from the tool including probabilities & reasoning.
- Placeholder RAG tool scaffold for future retrieval-augmented expansion.

## Project Structure
```
config/
  settings.py           # Central configuration & system prompt
utils/
  image_utils.py        # Image loading & preprocessing helpers
tools/
  sentiment_vit_tool.py # ViT emotion -> sentiment tool
agents/
  sentiment_agent.py    # Agent builder with system prompt and tools
rag/
  rag_placeholder.py    # Placeholder RAG tool
main.py                 # CLI entrypoint
requirements.txt        # Dependencies
README.md               # This file
```

## Installation
Use a virtual environment (recommended).

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> On first run the Hugging Face model weights will download (~hundreds of MB depending on model).

## Environment Variables
Set your OpenAI-compatible API key (example for OpenAI models):
```cmd
set OPENAI_API_KEY=sk-REPLACE_WITH_KEY
set OPENAI_MODEL=gpt-4o-mini  # optional, default already
```

### Groq (Primary Configuration)
The project is now configured to prioritize Groq.
Environment variable precedence:
1. `GROQ_API_KEY` (falls back to `API_KEY`)
2. `GROQ_MODEL` (falls back to `MODEL`, then default `llama-3.1-8b-instant`)
3. `GROQ_BASE_URL` (falls back to `API_BASE`, then `https://api.groq.com/openai/v1`)

Example setup:
```cmd
set GROQ_API_KEY=your_groq_key_here
set GROQ_MODEL=llama-3.1-8b-instant
rem Optional if Groq base changes
set GROQ_BASE_URL=https://api.groq.com/openai/v1
python main.py
```

If you previously saw `This method must be implemented in child classes`, it was due to using a generic model wrapper. The code now uses `OpenAIServerModel` directly with the Groq OpenAI-compatible endpoint.

## Running the Agent
```cmd
python main.py
```
Then interact:
```
> hello
> classify image ./examples/sample_face.png --notes "after a long day"
> classify image data:image/png;base64,iVBORw0KGgoAAA...
> help
> exit
```

## Tool Output Interpretation
The sentiment tool aggregates fine-grained emotion probabilities (e.g., happy, sad, angry) into coarse sentiment via a configurable mapping in `config/settings.py`.

Example JSON (pretty printed):
```json
{
  "sentiment": "positive",
  "emotions": {"happy": 0.78, "neutral": 0.12, "sad": 0.05},
  "dominant_emotion": "happy",
  "confidence": 0.90,
  "reasoning": "Dominant emotion: happy (0.78) | Aggregated sentiment scores: {'positive': 0.78, 'negative': 0.05, 'neutral': 0.12}"
}
```

## Future RAG Integration (Planned)
1. Add embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`).
2. Implement document ingestion pipeline (chunking, metadata storage).
3. Vector store (FAISS / Chroma) with similarity thresholding.
4. RAG tool returning top-k passages.
5. An answer synthesis tool that cites retrieved chunks.

## Extending / Improving
- Multi-face detection & per-face sentiment.
- Confidence calibration and threshold-based abstention.
- Additional sentiment categories (e.g., mixed, uncertain).
- Web UI (Streamlit/FastAPI) for drag-and-drop image upload.
- Caching model pipelines for faster cold starts (already lazy-loaded).
- Switch between alternative ViT emotion models via config.
- Add automated tests (pytest) for mapping and tool JSON schema.

## Streamlit App
A simple Streamlit UI is provided at `streamlit_app.py` for interactive use. Features:
- Chat-style conversation area.
- Image upload (saved to `data/uploads`) and immediate sentiment classification.
- Uses the full agent if `GROQ_API_KEY` is provided; otherwise runs the local ViT tool.

Run the Streamlit app:
```cmd
streamlit run streamlit_app.py
```

Uploaded images are saved under `data/uploads` relative to the project root. Ensure the process has write permissions to this directory.

## Notes on Accuracy & Ethics
Facial emotion recognition is inherently probabilistic and culturally sensitive. Do not use this system for high-stakes decisions. Always communicate uncertainty and avoid over-interpretation.

## License
No explicit license specified (add one if you intend to distribute).

---
Feel free to request enhancements or additional integrations.
