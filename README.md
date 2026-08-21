# Voice RAG — Low-Latency Voice-Enabled RAG Pipeline 🎙️⚡

> A high-performance, low-latency Voice-Enabled Retrieval-Augmented Generation (RAG) system with real-time procedural canvas audio waves, glassmorphism UI, 5 chunking strategies, and latency analytics.

---

## 🌟 Key Features

- **🎙️ Low-Latency Speech-to-Text**: Multi-provider support for **Sarvam AI** (`saaras:v1`), **ElevenLabs**, and high-speed **Mock Transcriber** for sub-200ms testing.
- **⚡ In-Memory Vector DB**: Low-latency cosine similarity vector search over the HuggingFace `ai4bharat/MSMARCO-XI` dataset using `sentence-transformers` (`all-MiniLM-L6-v2`).
- **🧠 5 Chunking Strategies**: Switchable on-the-fly:
  1. **Semantic / Sentence-Boundary** *(default)*
  2. **Fixed-Size with Sliding Window**
  3. **Metadata-Aware Structuring**
  4. **Hierarchical (Parent-Child)**
  5. **Recursive Multi-Delimiter**
- **🛡️ Multi-Stage Guardrails**: Input safety filtering and output grounding/hallucination checks.
- **📈 Latency Analytics & Benchmarking**: P50, P70, P90, P100 latency percentiles visualized with a custom Canvas2D **Polar Orbit Radar Chart** and mini sparklines.
- **🎨 Cinematic Glassmorphism UI**: Real-time procedural 28-layer audio-reactive ribbon waves, orbiting particles, floating glass orb, and responsive Q&A overlay.

---

## 🏗️ Tech Stack

| Backend | Frontend |
|---------|----------|
| Python 3.10+ / FastAPI | HTML5 / Vanilla CSS3 (Glassmorphism) |
| Uvicorn ASGI Server | HTML5 Canvas 2D API (Procedural Waves & Radar) |
| Sentence-Transformers (`all-MiniLM-L6-v2`) | Web Audio API (`MediaRecorder` + `AnalyserNode`) |
| NumPy / Scikit-Learn | Google Fonts (Outfit & JetBrains Mono) |
| Pydantic v2 / HTTPX | Vanilla JS (ES6+) |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn sentence-transformers scikit-learn numpy pydantic httpx datasets
```

### 4. (Optional) Set API Keys
```bash
# Windows PowerShell
$env:SARVAM_API_KEY="your_sarvam_key"
$env:ELEVENLABS_API_KEY="your_elevenlabs_key"

# Linux / macOS
export SARVAM_API_KEY="your_sarvam_key"
export ELEVENLABS_API_KEY="your_elevenlabs_key"
```

### 5. Run the Application
```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

Open your browser at `http://127.0.0.1:8000`.

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py                  # FastAPI app & API endpoints
│   ├── config.py                # App configuration & settings
│   ├── dataset_loader.py        # HuggingFace MSMARCO-XI loader
│   ├── stt/
│   │   └── stt_service.py       # Multi-provider STT (Sarvam / ElevenLabs / Mock)
│   ├── chunking/
│   │   ├── base.py              # Abstract chunker interface
│   │   └── strategies.py        # 5 chunking strategy implementations
│   ├── vector_db/
│   │   └── index.py             # Low-latency in-memory vector index
│   ├── guardrails/
│   │   └── guardrail_manager.py # Safety & grounding validation
│   ├── harness/
│   │   └── agent_harness.py     # 5-step pipeline orchestrator
│   └── analytics/
│       └── benchmark.py         # Latency percentile benchmark suite
├── frontend/
│   ├── index.html               # Single-page application shell
│   ├── css/styles.css           # Glassmorphism UI & responsive styles
│   └── js/app.js                # Canvas engine & Web Audio API
├── .gitignore
└── README.md
```

---

## 📊 API Reference

- `POST /api/voice-rag`: Main pipeline endpoint (accepts WAV audio or text query).
- `POST /api/benchmark`: Triggers 50-query latency percentile test suite (P50/P70/P90/P100).
- `GET /api/chunk-comparison`: Compares retrieval latency and chunks across all 5 strategies.
- `GET /api/health`: Service readiness & vector index status check.

---

## 📜 License

MIT License © 2026 Voice RAG Team.
