

```
## ⚠️ Runtime Notes

### 1. Avoid Using the `--reload` Flag on Windows
When running the backend server on Windows, do not start Uvicorn with the `--reload` flag. The reload option spawns separate worker processes that can interfere with file locks on the local in-memory FAISS index file. 

Start the server using this command:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000

```

### 2. Pydantic V2 Compatibility Note

The application backend uses Pydantic V2. To prevent method namespace collisions within Pydantic components, response handlers explicitly utilize `MCPResponse.make_success(...)` and `MCPResponse.make_error(...)`, outputting error details through a designated `error_detail` field.

---

## 🛠️ Tech Stack

* **Backend**: Python FastAPI, Sentence Transformers (`all-MiniLM-L6-v2`), FAISS Vector Database (`IndexFlatIP`), Local Ollama LLM (`tinyllama`), PyPDF2, and python-docx.
* **Frontend**: React (Vite), Tailwind CSS, Lucide React Icons, React Router DOM.
* **UI Design**: Minimalist and clean style using standard Tailwind CSS typography and card components.

---

## 📁 System Folder Structure

```
backend/
├── agent/
│   ├── __init__.py
│   ├── agent_controller.py
│   ├── planner.py
│   └── tool_router.py
├── tools/
│   ├── __init__.py
│   ├── retrieval_tool.py
│   ├── summarizer_tool.py
│   ├── upload_tool.py
│   ├── report_tool.py
│   └── list_tool.py
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── embeddings.py
│   ├── chunking.py
│   └── rag_pipeline.py
├── mcp/
│   ├── __init__.py
│   ├── tool_registry.py
│   ├── protocol.py
│   └── communication.py
├── api/
│   ├── __init__.py
│   └── routes.py
├── llm/
│   ├── __init__.py
│   ├── ollama_client.py
│   └── prompt_templates.py
├── data/
│   ├── sample_tech_spec.txt
│   ├── sample_research_findings.txt
│   └── sample_cybersecurity_assessment.txt
├── uploads/
├── logs/
├── models/
│   └── faiss_index/
├── config.py
├── main.py
├── seed_demo_data.py
└── requirements.txt
frontend/
├── src/
│   ├── components/
│   │   ├── Header.jsx
│   │   └── Sidebar.jsx
│   ├── pages/
│   │   ├── DashboardPage.jsx
│   │   ├── ChatPage.jsx
│   │   ├── UploadPage.jsx
│   │   ├── LogsPage.jsx
│   │   └── SettingsPage.jsx
│   ├── services/
│   │   └── api.js
│   ├── App.jsx
│   ├── index.css
│   └── main.jsx
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
└── postcss.config.js

```

---

## ⚡ Setup and Startup Instructions

Before starting the application, ensure that **Ollama** is installed and running locally on your machine:

```bash
ollama pull phi4-mini

```

### Terminal 1 — Backend Server

Navigate to the backend directory and set up your environment:

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000

```

### Terminal 2 — Frontend Development Server

Navigate to the frontend directory and start the web interface:

```bash
cd frontend

# Install Node modules
npm install

# Start Vite server
npm run dev

```

### Terminal 3 — Seed Demo Data (Optional)

To populate the FAISS database with initial text files for testing purposes, run the seeding script after the backend server is live:

```bash
cd backend
venv\Scripts\activate
python seed_demo_data.py

```

---

## IEEE Compliance Check

SIRS includes a built-in IEEE compliance validation layer integrated into the RAG pipeline.

### Supported Standards
| Standard   | Name                              |
|------------|-----------------------------------|
| IEEE 12207 | Software Life Cycle Processes     |
| IEEE 830   | Software Requirements Specifications |
| IEEE 829   | Software Test Documentation       |
| IEEE 1016  | Software Design Description       |
| IEEE 730   | Software Quality Assurance        |

### Endpoints
- `POST /compliance/check` — Run compliance check on any text
- `GET  /compliance/info`  — List all supported IEEE standards

### Running Tests
```bash
cd backend
python -m pytest tests/test_ieee_compliance.py -v
```

## 🧩 Application Core Tools

1. **retrieve_documents** (Action: `search`)
* Generates string embeddings using `all-MiniLM-L6-v2` and queries the FAISS index vector store.


2. **summarize_results** (Action: `summarize`)
* Consolidates retrieved documents into a concise response using `tinyllama`.


3. **upload_document** (Actions: `ingest`, `delete`)
* Parses incoming files, segments content into 512-character blocks with a 64-character overlap, generates embeddings, and saves them to the vector store. Deleting a file clears its segments and triggers an index rebuild.


4. **generate_report** (Action: `generate`)
* Creates structured, markdown-compatible textual reports from data inputs.


5. **list_documents** (Action: `list`)
* Exposes list arrays containing indexed filenames along with their total chunk calculations.



---

## 🛡️ Database and Sync Safety

* **vector_store.py** tracks state synchronization using a helper function called `_validate_sync()`. If the number of entries inside the FAISS index does not match the scalar length of your `metadata.pkl` file, the index will safely auto-truncate and rebuild to prevent crashes.
* Vector query bounds are safely checked inside the search functions to prevent unexpected out-of-bounds index errors from halting frontend components.

```
