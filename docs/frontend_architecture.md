# Frontend Architecture

The React frontend implements the screen transition diagram as a studio-style single page application. The onboarding screens use the provided login/splash background theme, while the authenticated studio uses a light top-menu layout.

## Screen Map

- `Splash` -> `Login`
- `Login` -> `Sign up` or `Main`
- `Sign up` -> `Login` or `Main`
- `Main` -> `Knowledge Bases`, `Evaluation`, `Analytics`
- `Main` assistant messages -> `Popup Source`, `Popup Trace`
- `Evaluation` -> `Evaluation Detail / RAGXplain`

## UI Composition

- Main screen has three columns:
  - Conversation History
  - Conversation / Chat
  - RAG Configuration
- Application navigation uses a top menu layout instead of a left side menu.
- Evaluation has Dataset and Evaluation panels.
- Analytics has Token statistics and Detailed Statistics & Feedbacks tabs.
- Chat calls FastAPI `POST /answer`.
- Feedback-ready UI calls FastAPI `POST /feedback`.
- Knowledge Bases calls FastAPI knowledge endpoints for live ingestion status, upload, website ingestion and re-indexing.

## Architecture Alignment

The UI mirrors the Architecture Landscape:

- Application layer: React RAG Studio and chatbot.
- Adaptive RAG: route decision, classifier, retriever, prompt assembly and response.
- Knowledge processing: source connectors, chunking, embedding and index management.
- Storage: vector DB, relational DB, chat history, metadata and usage logs.
- Model farm: DistilBERT, T5-small, embedding, reranker and generator models.
- Governance: RBAC, policy checks, citation validation and hallucination detection.
