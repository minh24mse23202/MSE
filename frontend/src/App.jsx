import { useEffect, useState } from "react";
import {
  askQuestion,
  createKnowledgeBase,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  getKnowledgeProcessingTrace,
  ingestWebsiteSource,
  listKnowledgeChunks,
  listKnowledgeDocuments,
  listKnowledgeBases,
  reindexKnowledgeBase,
  submitFeedback,
  updateKnowledgeBase,
  uploadKnowledgeSource
} from "./api.js";
import {
  architectureLayers,
  conversations,
  evaluationRuns,
  feedbackRows,
  seedMessages,
  tokenStats
} from "./data.js";

const navItems = [
  { id: "main", label: "Main" },
  { id: "knowledge", label: "Knowledge Bases" },
  { id: "evaluation", label: "Evaluation" },
  { id: "analytics", label: "Analytics" }
];

const classifiers = ["DistilBERT", "T5-small", "Naive Bayes", "Heuristic"];
const routes = ["Adaptive", "L1 Direct", "L2 Simple RAG", "L3 Complex RAG"];
const chunkingStrategies = [
  { value: "fixed_size", label: "Fixed-size Chunking" },
  { value: "sliding_window_overlap", label: "Sliding window / overlap chunking" },
  { value: "header_based", label: "Header-based Chunking" },
  { value: "semantic", label: "Semantic Chunking" },
  { value: "recursive", label: "Recursive Chunking" },
  { value: "hierarchical_parent_child", label: "Hierarchical / parent-child chunking" },
  { value: "structure_aware_custom", label: "Structure-aware / custom ara* chunking" }
];
const embeddingProviders = [
  "Local",
  "AI21",
  "Aleph Alpha",
  "Baidu",
  "Google",
  "Azure",
  "Cohere",
  "Fastembed",
  "Gradient",
  "Jina",
  "Mistral",
  "Voyage"
];
const embeddingModelsByProvider = {
  Local: ["hash-embedding-384", "sentence-transformers/all-MiniLM-L6-v2"],
  AI21: ["jamba-embeddings-v1"],
  "Aleph Alpha": ["luminous-base"],
  Baidu: ["bge-large-zh", "ernie-embedding"],
  Google: ["text-embedding-004", "gemini-embedding-001"],
  Azure: ["text-embedding-3-small", "text-embedding-3-large"],
  Cohere: ["embed-english-v3.0", "embed-multilingual-v3.0"],
  Fastembed: ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
  Gradient: ["bge-large"],
  Jina: ["jina-embeddings-v3", "jina-embeddings-v2-base-en"],
  Mistral: ["mistral-embed"],
  Voyage: ["voyage-3", "voyage-large-2"]
};
const supportedEmbeddingProvider = "Local";
const supportedLocalEmbeddingModels = embeddingModelsByProvider.Local;
const defaultKnowledgeConfiguration = {
  chunking_strategy: "sliding_window_overlap",
  chunk_size: 800,
  chunk_overlap: 120,
  embedding_provider: "Local",
  embedding_model: "hash-embedding-384"
};
const SPLASH_SEEN_KEY = "aragbiz:splash-seen";
const SPLASH_DURATION_MS = 1900;

export default function App() {
  const [screen, setScreen] = useState(() => (hasSeenSplash() ? "login" : "splash"));
  const [signedIn, setSignedIn] = useState(false);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState("");

  useEffect(() => {
    if (screen !== "splash") return undefined;
    const timer = window.setTimeout(() => {
      markSplashSeen();
      setScreen("login");
    }, SPLASH_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [screen]);

  function enterStudio() {
    setSignedIn(true);
    setScreen("main");
  }

  if (screen === "splash") {
    return <Splash />;
  }

  if (!signedIn && screen === "login") {
    return <AuthScreen mode="login" onSubmit={enterStudio} onSwitch={() => setScreen("signup")} />;
  }

  if (!signedIn && screen === "signup") {
    return <AuthScreen mode="signup" onSubmit={enterStudio} onSwitch={() => setScreen("login")} />;
  }

  return (
    <Shell activeScreen={screen} onNavigate={setScreen}>
      {screen === "main" && (
        <MainScreen
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
        />
      )}
      {screen === "knowledge" && (
        <KnowledgeBasesScreen
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
        />
      )}
      {screen === "evaluation" && <EvaluationScreen onOpenDetail={() => setScreen("evaluation-detail")} />}
      {screen === "evaluation-detail" && <EvaluationDetailScreen />}
      {screen === "analytics" && <AnalyticsScreen />}
    </Shell>
  );
}

function Splash() {
  return (
    <main className="splash login-screen">
      <section className="splash-panel login-form">
        <p className="eyebrow">Adaptive RAG System</p>
        <h1>Business Workflow QA Studio</h1>
        <p>
          Preparing classifiers, retrieval routes, trace views and evaluation panels.
        </p>
        <div className="splash-loader splash-loading" aria-label="Loading Adaptive RAG Studio">
          <span className="splash-loading-bar" />
        </div>
        <p className="splash-status">Initializing studio workspace...</p>
        <div className="architecture-strip">
          {architectureLayers.slice(0, 4).map((layer) => (
            <span key={layer}>{layer.split(":")[0]}</span>
          ))}
        </div>
      </section>
    </main>
  );
}

function AuthScreen({ mode, onSubmit, onSwitch }) {
  const isSignup = mode === "signup";
  if (isSignup) {
    return <SignupScreen onSubmit={onSubmit} onSwitch={onSwitch} />;
  }
  return (
    <main className="auth-layout login-screen">
      <section className="auth-card login-form">
        <div className="auth-logo">
          <span className="logo-dot" />
          <strong className="eyebrow">Adaptive RAG Studio</strong>
        </div>
        <h1>Login</h1>
        <label>
          Email
          <input defaultValue="quangminhrt@gmail.com" type="email" />
        </label>
        <label>
          Password
          <input defaultValue="adaptive-rag" type="password" />
        </label>
        <button className="primary-action" onClick={onSubmit}>Login</button>
        <button className="text-action" onClick={onSwitch}>
          Need an account? Sign up
        </button>
      </section>
    </main>
  );
}

function SignupScreen({ onSubmit, onSwitch }) {
  return (
    <main className="auth-layout signup-layout login-screen">
      <section className="signup-card login-form">
        <div className="auth-logo">
          <span className="logo-dot" />
          <strong>Business Worklow Question Answering</strong>
        </div>
        <h1>Sign up new account</h1>
        <div className="signup-row">
          <label>
            First name
            <input type="text" autoComplete="given-name" />
          </label>
          <label>
            Last name
            <input type="text" autoComplete="family-name" />
          </label>
        </div>
        <label>
          Email
          <input type="email" autoComplete="email" />
        </label>
        <label>
          Password
          <input type="password" autoComplete="new-password" />
        </label>
        <div className="recaptcha-box" aria-label="reCAPTCHA verification placeholder">
          <label className="captcha-check">
            <input type="checkbox" />
            <span>I'm not a robot</span>
          </label>
          <div className="captcha-brand">
            <span>reload</span>
            <small>reCAPTCHA</small>
          </div>
        </div>
        <label className="signup-consent">
          <input type="checkbox" />
          <span>
            I agree to use Adaptive RAG Studio as an AI‑powered system. I will verify answers since AI can make mistakes.
          </span>
        </label>
        <button className="primary-action signup-submit" onClick={onSubmit}>Agree and start</button>
        <button className="text-action signup-login-link" onClick={onSwitch}>
          Already have an account? Log in
        </button>
      </section>
    </main>
  );
}

function Shell({ activeScreen, onNavigate, children }) {
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">AR</span>
          <div>
            <strong>Adaptive RAG</strong>
            <small>Workflow QA Studio</small>
          </div>
        </div>
        <nav>
          {navItems.map((item) => (
            <button
              key={item.id}
              className={activeScreen === item.id ? "active" : ""}
              onClick={() => onNavigate(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <div className="topbar-actions">
          <span className="user-chip">Researcher</span>
        </div>
      </header>
      <main className="workspace">{children}</main>
    </div>
  );
}

function MainScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase }) {
  const [messages, setMessages] = useState(seedMessages);
  const [question, setQuestion] = useState("Can I start accepting payments while Wix Payments is under verification?");
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [popup, setPopup] = useState(null);
  const [knowledgeBaseOptions, setKnowledgeBaseOptions] = useState([]);
  const [config, setConfig] = useState({
    classifier: "DistilBERT",
    route: "Adaptive",
    retrievalMode: "Hybrid",
    topK: 6,
    reranker: true,
    citations: true
  });

  useEffect(() => {
    let isMounted = true;
    listKnowledgeBases()
      .then((items) => {
        if (isMounted) setKnowledgeBaseOptions(items);
      })
      .catch(() => {
        if (isMounted) setKnowledgeBaseOptions([]);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  async function sendQuestion() {
    const trimmed = question.trim();
    if (!trimmed) return;
    const userMessage = { id: createId(), role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsLoading(true);
    try {
      const response = await askQuestion(trimmed, selectedKnowledgeBaseId);
      setMessages((current) => [
        ...current,
        {
          id: createId(),
          question: trimmed,
          role: "assistant",
          content: response.answer,
          contexts: response.contexts,
          metadata: response.metadata
        }
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: createId(),
          question: trimmed,
          role: "assistant",
          content: "The API is not reachable. Start FastAPI on port 8000, then retry.",
          contexts: [],
          metadata: { error: error.message, complexity_label: "unknown" }
        }
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  async function recordFeedback(message, rating) {
    setFeedbackStatus("Recording feedback...");
    try {
      await submitFeedback({
        question: message.question || "Seed assistant message",
        answer: message.content,
        rating,
        metadata: message.metadata || {}
      });
      setFeedbackStatus("Feedback recorded");
    } catch (error) {
      setFeedbackStatus(`Feedback failed: ${error.message}`);
    }
  }

  return (
    <section className="main-grid">
      <ConversationHistory />
      <ChatPanel
        messages={messages}
        question={question}
        setQuestion={setQuestion}
        isLoading={isLoading}
        onSend={sendQuestion}
        onOpenPopup={setPopup}
        onFeedback={recordFeedback}
        knowledgeBases={knowledgeBaseOptions}
        selectedKnowledgeBaseId={selectedKnowledgeBaseId}
        onSelectKnowledgeBase={onSelectKnowledgeBase}
      />
      <RagConfiguration
        config={config}
        setConfig={setConfig}
        selectedKnowledgeBase={knowledgeBaseOptions.find((item) => item.id === selectedKnowledgeBaseId)}
      />
      {feedbackStatus && <div className="toast">{feedbackStatus}</div>}
      {popup && <TraceModal popup={popup} onClose={() => setPopup(null)} />}
    </section>
  );
}

function ConversationHistory() {
  return (
    <aside className="history-panel">
      <button className="back-link" type="button">Back to Brains</button>
      <button className="new-chat-action" type="button">New chat</button>
      <p className="history-period">June</p>
      <div className="conversation-list">
        {conversations.map((conversation) => (
          <button
            key={conversation.id}
            className={`conversation-item ${conversation.status === "active" ? "active" : ""}`}
            type="button"
          >
            <span>
              <strong>{conversation.title}</strong>
              <small>{conversation.route}</small>
            </span>
            <em>{conversation.updatedAt}</em>
          </button>
        ))}
      </div>
    </aside>
  );
}

function ChatPanel({
  messages,
  question,
  setQuestion,
  isLoading,
  onSend,
  onOpenPopup,
  onFeedback,
  knowledgeBases,
  selectedKnowledgeBaseId,
  onSelectKnowledgeBase
}) {
  return (
    <section className="chat-panel">
      <header className="chat-titlebar">
        <div className="brain-mark">AI</div>
        <div>
          <h1>Business Workflow Question Answering</h1>
          <p>Business Workflow Question Answering AI Chatbot</p>
        </div>
      </header>
      <div className="message-list">
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            {message.role === "assistant" && <span className="avatar">AR</span>}
            <div>
              <p>{message.content}</p>
            </div>
            {message.role === "assistant" && (
              <div className="message-actions">
                <button onClick={() => onOpenPopup({ type: "source", message })}>Sources</button>
                <button type="button">Copy</button>
                <button onClick={() => onOpenPopup({ type: "trace", message })}>Trace</button>
                <button onClick={() => onFeedback(message, "up")}>Useful</button>
                <button onClick={() => onFeedback(message, "down")}>Needs work</button>
                <span>{message.metadata?.complexity_label || "pending"}</span>
              </div>
            )}
          </article>
        ))}
        {isLoading && <article className="message assistant"><p>Routing query and retrieving context...</p></article>}
      </div>
      <div className="composer">
        <textarea
          aria-label="Chat message"
          placeholder='Send a message to brain "Wix Chatbot"'
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <div className="composer-tools">
          <span>Attach</span>
          <span>Filter</span>
          <span>0%</span>
          <select
            className="composer-select"
            aria-label="Select knowledge base"
            value={selectedKnowledgeBaseId}
            onChange={(event) => onSelectKnowledgeBase(event.target.value)}
          >
            <option value="">Select Knowledge Base</option>
            {knowledgeBases.map((knowledgeBase) => (
              <option key={knowledgeBase.id} value={knowledgeBase.id}>
                {knowledgeBase.name}
              </option>
            ))}
          </select>
          <strong>RAG mode</strong>
          <button className="send-action" onClick={onSend}>Send</button>
        </div>
      </div>
      <p className="ai-disclaimer">This is an AI-powered system. Please verify answers since AI can make mistakes.</p>
    </section>
  );
}

function RagConfiguration({ config, setConfig, selectedKnowledgeBase }) {
  return (
    <aside className="config-panel">
      <div className="config-topbar">
        <h2>Configuration</h2>
        <div>
          <button type="button">Data security</button>
          <button type="button">Share</button>
          <button type="button" disabled>Save changes</button>
        </div>
      </div>
      <section className="config-section">
        <header>
          <h3>General</h3>
          <button type="button">Collapse</button>
        </header>
        <label>
          Name
          <small>Choose a name that clearly represents the purpose of your brain.</small>
          <input defaultValue="Business Workflow Question Answering" />
        </label>
        <label>
          Description
          <small>Describe your brain and share basic information that explains its capabilities.</small>
          <textarea defaultValue="Business Workflow Question Answering AI Chatbot" />
        </label>
        <label>
          Functional area
          <small>Select the area and domain in which your brain will be used.</small>
          <div className="tag-cloud">
            {["Human resources", "Information Coordination Organisation", "Quality Management", "General Functions", "Logistics"].map((tag) => (
              <span key={tag}>{tag}</span>
            ))}
          </div>
        </label>
        <div className="icon-grid" aria-label="Brain icon choices">
          {["AI", "KB", "QA", "TR", "RT", "EV", "HD", "TM", "AN"].map((icon, index) => (
            <button key={icon} className={index === 0 ? "selected" : ""} type="button">{icon}</button>
          ))}
        </div>
      </section>
      <section className="config-section">
        <header>
          <h3>Inputs</h3>
          <button type="button">Collapse</button>
        </header>
        <SelectField
          label="Classifier"
          value={config.classifier}
          options={classifiers}
          onChange={(classifier) => setConfig({ ...config, classifier })}
        />
        <SelectField
          label="Route strategy"
          value={config.route}
          options={routes}
          onChange={(route) => setConfig({ ...config, route })}
        />
        <SelectField
          label="Retrieval mode"
          value={config.retrievalMode}
          options={["Hybrid", "BM25", "Dense"]}
          onChange={(retrievalMode) => setConfig({ ...config, retrievalMode })}
        />
        <label>
          Top K retrieval
          <input
            type="range"
            min="1"
            max="12"
            value={config.topK}
            onChange={(event) => setConfig({ ...config, topK: Number(event.target.value) })}
          />
          <strong>{config.topK} contexts</strong>
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={config.reranker}
            onChange={(event) => setConfig({ ...config, reranker: event.target.checked })}
          />
          Enable reranker
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={config.citations}
            onChange={(event) => setConfig({ ...config, citations: event.target.checked })}
          />
          Citation validator
        </label>
        <div className="route-card">
          <strong>Selected knowledge base</strong>
          <span>{selectedKnowledgeBase ? `${selectedKnowledgeBase.name} (${selectedKnowledgeBase.document_count} docs, ${selectedKnowledgeBase.chunk_count} chunks)` : "No knowledge base selected"}</span>
        </div>
      </section>
    </aside>
  );
}

function TraceModal({ popup, onClose }) {
  const { type, message } = popup;
  const contexts = message.contexts || [];
  const sourceContexts = contexts.length > 0 ? contexts : [
    {
      id: "wix-uac-001",
      rank: 1,
      score: 0.87,
      text: "UAT is performed before go-live to validate the migrated application and confirm readiness for production. Standard duration is approximately two weeks with possible extension."
    },
    {
      id: "wix-uac-002",
      rank: 2,
      score: 0.82,
      text: "For specific contacts, batch sequence, and timelines, application teams should check the migration dashboard and the WorkON support documentation."
    }
  ];
  const traceSteps = [
    ["Prompt builder", "0ms", "System role, scope, response policy, and citation rules were assembled."],
    ["Chat input", `${message.metadata?.latency_ms || 396}ms`, message.question || "User question captured from the chat composer."],
    ["Query complexity classifier", "42ms", `Predicted ${message.metadata?.complexity_label || "moderate"} complexity.`],
    ["Hybrid retrieval", "310ms", `Mode ${message.metadata?.retrieval_mode || "hybrid"} with top K ${message.metadata?.top_k || sourceContexts.length}.`],
    ["Generator", "1.8s", "Generated grounded answer using retrieved workflow passages."]
  ];
  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className={`modal ${type === "source" ? "source-modal" : "trace-modal"}`} role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <h2>{type === "source" ? "Source Documents" : "Trace report"}</h2>
            <p>{type === "source" ? "Review the passages used to ground this answer." : "A detailed report of the chat pipeline execution and outcomes."}</p>
          </div>
          <button className="icon-button modal-close" aria-label="Close modal" onClick={onClose}>x</button>
        </header>
        {type === "source" ? (
          <div className="source-modal-grid">
            <div className="source-sidebar">
              {sourceContexts.map((context, index) => (
                <button key={context.id} className={`source-result ${index === 0 ? "selected" : ""}`} type="button">
                  <span>File</span>
                  <strong>Business Workflow QA source {context.rank}</strong>
                  <small>{index % 2 === 0 ? "Semantic Search" : "Full Text Search"}</small>
                  <em>{Math.round(Number(context.score || 0.8) * 100)}% match</em>
                </button>
              ))}
            </div>
            <article className="source-preview">
              <a href="#top">Business Workflow Question Answering Source.docx</a>
              <small>Search type: Semantic Search</small>
              <p>
                Title: Business Workflow Question Answering Source.docx<br />
                Content: Workflow support documents provide detailed process information, role responsibilities, timeline guidance, and escalation notes.
              </p>
              {sourceContexts.map((context) => (
                <p key={context.id}>{context.text}</p>
              ))}
            </article>
          </div>
        ) : (
          <div className="trace-layout">
            <aside className="trace-steps">
              <h3>Steps Overview</h3>
              {traceSteps.map(([name], index) => (
                <button key={name} className={index === 0 ? "active" : ""} type="button">{index + 1}. {name}</button>
              ))}
            </aside>
            <div className="trace-content">
              <label className="trace-search">
                <input placeholder="Search in trace report" />
              </label>
              {traceSteps.map(([name, duration, body]) => (
                <article key={name} className="trace-step-card">
                  <header>
                    <h3>{name} <span>{duration}</span></h3>
                    <button type="button">Copy</button>
                  </header>
                  <pre>{`NodeId: ${createId()}
NodeName: ${name}
NodeType: ${name.includes("retrieval") ? "Retriever" : "Pipeline step"}
Timestamp: 2026-06-27T09:00:00.000Z

${body}`}</pre>
                </article>
              ))}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function KnowledgeBasesScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase }) {
  const [items, setItems] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [processingTrace, setProcessingTrace] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingDocuments, setIsLoadingDocuments] = useState(false);
  const [error, setError] = useState("");
  const [knowledgeBaseModal, setKnowledgeBaseModal] = useState(null);
  const [actionStatus, setActionStatus] = useState("");
  const [detailTab, setDetailTab] = useState("documents");

  async function refreshKnowledgeBases() {
    setIsLoading(true);
    setError("");
    try {
      const nextItems = await listKnowledgeBases();
      setItems(nextItems);
      if (!selectedKnowledgeBaseId && nextItems.length > 0) {
        onSelectKnowledgeBase(nextItems[0].id);
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  async function refreshDocuments(knowledgeBaseId = selectedKnowledgeBaseId) {
    if (!knowledgeBaseId) {
      setDocuments([]);
      setChunks([]);
      setProcessingTrace([]);
      return;
    }
    setIsLoadingDocuments(true);
    try {
      const [nextDocuments, nextChunks, nextTrace] = await Promise.all([
        listKnowledgeDocuments(knowledgeBaseId),
        listKnowledgeChunks(knowledgeBaseId),
        getKnowledgeProcessingTrace(knowledgeBaseId)
      ]);
      setDocuments(nextDocuments);
      setChunks(nextChunks);
      setProcessingTrace(nextTrace);
    } catch (requestError) {
      setActionStatus(`Document load failed: ${requestError.message}`);
    } finally {
      setIsLoadingDocuments(false);
    }
  }

  useEffect(() => {
    refreshKnowledgeBases();
  }, []);

  useEffect(() => {
    refreshDocuments();
  }, [selectedKnowledgeBaseId]);

  async function handleReindex(knowledgeBaseId) {
    setActionStatus("Re-indexing knowledge base...");
    try {
      await reindexKnowledgeBase(knowledgeBaseId);
      setActionStatus("Re-index completed");
      await refreshKnowledgeBases();
      await refreshDocuments(knowledgeBaseId);
    } catch (requestError) {
      setActionStatus(`Re-index failed: ${requestError.message}`);
    }
  }

  async function handleDeleteDocument(documentId) {
    if (!selectedKnowledgeBaseId) return;
    const document = documents.find((item) => item.id === documentId);
    const confirmed = window.confirm(`Delete "${document?.title || "this document"}" and all of its chunks?`);
    if (!confirmed) return;
    setActionStatus("Deleting document...");
    try {
      await deleteKnowledgeDocument(selectedKnowledgeBaseId, documentId);
      setActionStatus("Document deleted");
      await refreshKnowledgeBases();
      await refreshDocuments(selectedKnowledgeBaseId);
    } catch (requestError) {
      setActionStatus(`Delete failed: ${requestError.message}`);
    }
  }

  async function handleDeleteKnowledgeBase() {
    if (!selectedKnowledgeBase) return;
    const confirmed = window.confirm(`Delete "${selectedKnowledgeBase.name}" and all of its documents?`);
    if (!confirmed) return;
    setActionStatus("Deleting knowledge base and all documents...");
    try {
      await deleteKnowledgeBase(selectedKnowledgeBase.id);
      setActionStatus("Knowledge base deleted");
      onSelectKnowledgeBase("");
      setDocuments([]);
      setChunks([]);
      setProcessingTrace([]);
      await refreshKnowledgeBases();
    } catch (requestError) {
      setActionStatus(`Delete knowledge base failed: ${requestError.message}`);
    }
  }

  const selectedKnowledgeBase = items.find((item) => item.id === selectedKnowledgeBaseId);
  const chunksByDocument = chunks.reduce((groups, chunk) => {
    const nextGroups = groups;
    if (!nextGroups[chunk.document_id]) nextGroups[chunk.document_id] = [];
    nextGroups[chunk.document_id].push(chunk);
    return nextGroups;
  }, {});

  return (
    <section className="page-stack">
      <PanelHeader eyebrow="Knowledge base storage" title="Knowledge Bases" />
      <div className="toolbar">
        <button className="primary-action" onClick={() => setKnowledgeBaseModal({ mode: "create" })}>Add knowledge base</button>
        <button className="secondary-action" onClick={refreshKnowledgeBases}>Refresh</button>
      </div>
      {error && <div className="alert">Knowledge API unavailable: {error}</div>}
      {actionStatus && <div className="inline-status">{actionStatus}</div>}
      <div className="table-panel">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Description</th>
              <th>Documents</th>
              <th>Chunks</th>
              <th>Embedding model</th>
              <th>Status</th>
              <th>Last indexed</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && (
              <tr>
                <td colSpan="7">Loading knowledge bases...</td>
              </tr>
            )}
            {!isLoading && items.length === 0 && (
              <tr>
                <td colSpan="7">No knowledge bases yet. Add one from local files or a public website.</td>
              </tr>
            )}
            {items.map((kb) => (
              <tr
                key={kb.id}
                className={`selectable-row ${kb.id === selectedKnowledgeBaseId ? "selected-row" : ""}`}
                tabIndex={0}
                aria-selected={kb.id === selectedKnowledgeBaseId}
                onClick={() => onSelectKnowledgeBase(kb.id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectKnowledgeBase(kb.id);
                  }
                }}
              >
                <td>{kb.name}</td>
                <td>{kb.description || "No description"}</td>
                <td>{kb.document_count}</td>
                <td>{kb.chunk_count}</td>
                <td>{kb.embedding_model || "Not indexed"}</td>
                <td>
                  <span className={`status-pill status-${kb.status}`}>{kb.status}</span>
                  {kb.error && <small className="error-text">{kb.error}</small>}
                </td>
                <td>{formatDateTime(kb.updated_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <section className="document-manager">
        <div className="document-manager-header">
          <div>
            <p className="eyebrow">Documents</p>
            <h2>{selectedKnowledgeBase ? selectedKnowledgeBase.name : "Select a knowledge base"}</h2>
          </div>
          {selectedKnowledgeBase && (
            <div className="document-manager-actions">
              <button className="secondary-action" onClick={() => setKnowledgeBaseModal({ mode: "edit", knowledgeBase: selectedKnowledgeBase })}>
                Modify KB
              </button>
              <button className="secondary-action" onClick={() => handleReindex(selectedKnowledgeBase.id)}>
                Re-index
              </button>
              <button className="secondary-action danger-action" onClick={handleDeleteKnowledgeBase}>
                Delete KB
              </button>
            </div>
          )}
        </div>
        {!selectedKnowledgeBase && <p className="muted-text">Choose a knowledge base above to add, modify, or delete documents.</p>}
        {selectedKnowledgeBase && (
          <>
            <div className="detail-tabs" role="tablist" aria-label="Knowledge base details">
              <button
                className={detailTab === "documents" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={detailTab === "documents"}
                onClick={() => setDetailTab("documents")}
              >
                Documents
              </button>
              <button
                className={detailTab === "trace" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={detailTab === "trace"}
                onClick={() => setDetailTab("trace")}
              >
                Processing trace
              </button>
            </div>
            {detailTab === "documents" ? (
              <div className="document-list detail-tab-panel">
                {isLoadingDocuments && <p className="muted-text">Loading documents...</p>}
                {!isLoadingDocuments && documents.length === 0 && <p className="muted-text">No documents yet. Use Modify KB to add local files or website sources.</p>}
                {documents.map((document, index) => (
                  <details key={document.id} className="document-card document-accordion" defaultOpen={index === 0}>
                    <summary className="document-card-top">
                      <div>
                        <strong>{document.title}</strong>
                        <small>{document.metadata?.source_type || "document"} - {document.text.length} chars - {document.content_hash.slice(0, 10)}</small>
                      </div>
                      <button
                        className="secondary-action compact-action danger-action"
                        type="button"
                        onClick={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          handleDeleteDocument(document.id);
                        }}
                      >
                        Delete document
                      </button>
                    </summary>
                    <div className="document-accordion-body">
                      <p className="document-preview">{document.text.slice(0, 260)}{document.text.length > 260 ? "..." : ""}</p>
                      <MetadataGrid metadata={document.metadata} />
                      <DocumentChunkList chunks={chunksByDocument[document.id] || []} />
                    </div>
                  </details>
                ))}
              </div>
            ) : (
              <div className="detail-tab-panel">
                <ProcessingTracePanel steps={processingTrace} />
              </div>
            )}
          </>
        )}
      </section>
      {knowledgeBaseModal && (
        <AddKnowledgeBaseModal
          mode={knowledgeBaseModal.mode}
          knowledgeBase={knowledgeBaseModal.knowledgeBase}
          onClose={() => setKnowledgeBaseModal(null)}
          onCreated={async () => {
            setKnowledgeBaseModal(null);
            await refreshKnowledgeBases();
            await refreshDocuments();
          }}
        />
      )}
    </section>
  );
}

function MetadataGrid({ metadata = {} }) {
  const entries = Object.entries(metadata || {});
  if (entries.length === 0) {
    return <p className="muted-text compact-muted">No metadata captured yet.</p>;
  }
  return (
    <dl className="metadata-grid">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key}</dt>
          <dd>{formatMetadataValue(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function DocumentChunkList({ chunks = [] }) {
  return (
    <section className="chunk-section">
      <div className="chunk-section-header">
        <strong>Chunks</strong>
        <span>{chunks.length} chunk{chunks.length === 1 ? "" : "s"}</span>
      </div>
      {chunks.length === 0 ? (
        <p className="muted-text compact-muted">No chunks indexed for this document.</p>
      ) : (
        <div className="chunk-list">
          {chunks.map((chunk) => (
            <article key={chunk.id} className="chunk-card">
              <header>
                <div>
                  <strong>Chunk {chunk.chunk_index + 1}</strong>
                  <small>{chunk.token_count} tokens - {chunk.text.length} chars</small>
                </div>
                <span className={`status-pill ${chunk.has_embedding ? "status-indexed" : "status-waiting"}`}>
                  {chunk.has_embedding ? "embedded" : "not embedded"}
                </span>
              </header>
              <dl className="chunk-facts">
                <div>
                  <dt>Embedding model</dt>
                  <dd>{chunk.embedding_model || "Not available"}</dd>
                </div>
                <div>
                  <dt>Vector dimension</dt>
                  <dd>{chunk.embedding_dimension || 0}</dd>
                </div>
                <div>
                  <dt>Range</dt>
                  <dd>{chunk.metadata?.start_char ?? "-"} - {chunk.metadata?.end_char ?? "-"}</dd>
                </div>
                <div>
                  <dt>Mode</dt>
                  <dd>{chunk.metadata?.chunking_mode || "overlap"}</dd>
                </div>
              </dl>
              <pre>{chunk.text}</pre>
              <MetadataGrid metadata={chunk.metadata} />
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function ProcessingTracePanel({ steps = [] }) {
  return (
    <section className="processing-trace-panel">
      <div className="processing-trace-header">
        <div>
          <p className="eyebrow">Processing trace</p>
          <h3>Knowledge and data processing pipeline</h3>
        </div>
        <span>{steps.length} steps</span>
      </div>
      {steps.length === 0 ? (
        <p className="muted-text">No processing trace available yet.</p>
      ) : (
        <div className="processing-trace-list">
          {steps.map((step, index) => (
            <article key={`${step.step}-${index}`} className="processing-trace-step">
              <span className="trace-step-index">{index + 1}</span>
              <div>
                <header>
                  <strong>{step.step}</strong>
                  <span className={`status-pill status-${step.status}`}>{step.status}</span>
                </header>
                <p>{step.detail}</p>
                {(step.started_at || step.finished_at) && (
                  <small>
                    {step.started_at ? `Started ${formatDateTime(step.started_at)}` : ""}
                    {step.finished_at ? ` - Finished ${formatDateTime(step.finished_at)}` : ""}
                  </small>
                )}
                <details>
                  <summary>Metadata</summary>
                  <MetadataGrid metadata={step.metadata} />
                </details>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

function AddKnowledgeBaseModal({
  mode = "create",
  knowledgeBase,
  onClose,
  onCreated
}) {
  const isEdit = mode === "edit";
  const [name, setName] = useState(knowledgeBase?.name || "Business Workflow Knowledge Base");
  const [description, setDescription] = useState(knowledgeBase?.description || "Uploaded workflow documents and website content.");
  const [configuration, setConfiguration] = useState(() => knowledgeConfigurationFromRecord(knowledgeBase));
  const [sourceType, setSourceType] = useState("upload");
  const [files, setFiles] = useState([]);
  const [websiteUrl, setWebsiteUrl] = useState("");
  const [status, setStatus] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setIsSubmitting(true);
    setStatus(isEdit ? "Updating knowledge base..." : "Creating knowledge base...");
    try {
      const payload = {
        name,
        description,
        configuration: sanitizeKnowledgeConfiguration(configuration)
      };
      const savedKnowledgeBase = isEdit
        ? await updateKnowledgeBase(knowledgeBase.id, payload)
        : await createKnowledgeBase(payload);
      if (sourceType === "upload") {
        if (!isEdit && !files.length) throw new Error("Choose at least one local file.");
        if (files.length) {
          setStatus("Uploading and indexing files...");
          await uploadKnowledgeSource(savedKnowledgeBase.id, files);
        }
      } else if (websiteUrl.trim()) {
        setStatus("Reading website and indexing content...");
        await ingestWebsiteSource(savedKnowledgeBase.id, websiteUrl.trim());
      } else if (!isEdit) {
        throw new Error("Enter a public website URL.");
      }
      setStatus(isEdit ? "Knowledge base updated" : "Knowledge base indexed");
      await onCreated();
    } catch (requestError) {
      setStatus(requestError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="modal kb-modal" role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <h2>{isEdit ? "Modify knowledge base" : "Add knowledge base"}</h2>
            <p>{isEdit ? "Update name, description, add more documents, or delete existing documents." : "Enter details and select multiple files or a public website source."}</p>
          </div>
          <button className="icon-button modal-close" aria-label="Close modal" onClick={onClose}>x</button>
        </header>
        <form className="kb-form" onSubmit={submit}>
          <label>
            Knowledge base name
            <input value={name} onChange={(event) => setName(event.target.value)} />
          </label>
          <label>
            Description
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <section className="kb-config-section">
            <div>
              <h3>Configuration</h3>
              <p>These settings are saved with the knowledge base and applied to new ingestion or re-indexing.</p>
            </div>
            <label>
              Chunking strategy
              <select
                value={configuration.chunking_strategy}
                onChange={(event) => {
                  const strategy = event.target.value;
                  setConfiguration((current) => ({
                    ...current,
                    chunking_strategy: strategy,
                    chunk_overlap: strategy === "fixed_size" ? 0 : current.chunk_overlap
                  }));
                }}
              >
                {chunkingStrategies.map((strategy) => (
                  <option key={strategy.value} value={strategy.value}>{strategy.label}</option>
                ))}
              </select>
            </label>
            <div className="kb-config-grid">
              <label>
                Chunk size
                <input
                  type="number"
                  min="100"
                  max="12000"
                  step="50"
                  value={configuration.chunk_size}
                  onChange={(event) => setConfiguration((current) => ({ ...current, chunk_size: Number(event.target.value) }))}
                />
              </label>
              <label>
                Overlap size
                <input
                  type="number"
                  min="0"
                  max={Math.max(Number(configuration.chunk_size) - 1, 0)}
                  step="10"
                  disabled={!chunkingStrategyUsesOverlap(configuration.chunking_strategy)}
                  value={chunkingStrategyUsesOverlap(configuration.chunking_strategy) ? configuration.chunk_overlap : 0}
                  onChange={(event) => setConfiguration((current) => ({ ...current, chunk_overlap: Number(event.target.value) }))}
                />
              </label>
            </div>
            <div className="kb-config-grid">
              <label>
                Embedding provider
                <select
                  value={configuration.embedding_provider}
                  onChange={(event) => {
                    const provider = event.target.value === supportedEmbeddingProvider
                      ? supportedEmbeddingProvider
                      : defaultKnowledgeConfiguration.embedding_provider;
                    setConfiguration((current) => ({
                      ...current,
                      embedding_provider: provider,
                      embedding_model: supportedLocalEmbeddingModels[0] || current.embedding_model
                    }));
                  }}
                >
                  {embeddingProviders.map((provider) => (
                    <option key={provider} value={provider} disabled={provider !== supportedEmbeddingProvider}>
                      {provider}{provider !== supportedEmbeddingProvider ? " (coming soon)" : ""}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Embedding model
                <select
                  value={configuration.embedding_model}
                  onChange={(event) => setConfiguration((current) => ({ ...current, embedding_model: event.target.value }))}
                >
                  {supportedLocalEmbeddingModels.map((model) => (
                    <option key={model} value={model}>{model}</option>
                  ))}
                </select>
              </label>
            </div>
            <p className="muted-text compact-muted">
              V1 executes Local embeddings only. `hash-embedding-384` works with the base API install; MiniLM requires `python -m pip install -e ".[ml]"`.
              External providers stay visible for the roadmap and are disabled until provider adapters and API-key handling are added.
            </p>
          </section>
          <div className="source-selector">
            <button type="button" className={sourceType === "upload" ? "selected" : ""} onClick={() => setSourceType("upload")}>
              {isEdit ? "Add local files" : "Local device"}
            </button>
            <button type="button" className={sourceType === "website" ? "selected" : ""} onClick={() => setSourceType("website")}>
              {isEdit ? "Add website" : "Public website"}
            </button>
            <button type="button" disabled>Google Drive soon</button>
            <button type="button" disabled>OneDrive soon</button>
            <button type="button" disabled>Databases soon</button>
          </div>
          {sourceType === "upload" ? (
            <label>
              Files
              <input
                type="file"
                multiple
                accept=".txt,.json,.jsonl,.md,.pdf,.docx,.aratxt,.arajson,.aramd"
                onChange={(event) => setFiles(Array.from(event.target.files || []))}
              />
              <small>Supported: txt, json, jsonl, md, pdf, docx, aratxt, arajson, aramd.</small>
            </label>
          ) : (
            <label>
              Public website URL
              <input
                value={websiteUrl}
                placeholder="https://example.com/workflow-guide"
                onChange={(event) => setWebsiteUrl(event.target.value)}
              />
            </label>
          )}
          <div className="pipeline-preview">
            {["Metadata extraction", "Deduplication", "Chunking", "Embedding", "PostgreSQL + pgVector"].map((step) => (
              <span key={step}>{step}</span>
            ))}
          </div>
          {status && <p className="inline-status">{status}</p>}
          <div className="modal-actions">
            <button type="button" className="secondary-action" onClick={onClose}>Cancel</button>
            <button type="submit" className="primary-action" disabled={isSubmitting}>
              {isSubmitting ? "Processing..." : isEdit ? "Save knowledge base" : "Create and index"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function EvaluationScreen({ onOpenDetail }) {
  return (
    <section className="evaluation-grid">
      <section className="panel">
        <PanelHeader eyebrow="Panel" title="Dataset" />
        <Metric label="Dataset" value="WixQA expert-written" />
        <Metric label="Records" value="200" />
        <Metric label="KB documents" value="6,221" />
        <Metric label="Complexity labels" value="simple / moderate / complex" />
      </section>
      <section className="panel">
        <PanelHeader eyebrow="Panel" title="Evaluation" />
        <div className="run-list">
          {evaluationRuns.map((run) => (
            <article key={run.id} className="run-card">
              <div>
                <strong>{run.dataset}</strong>
                <small>{run.classifier} - {run.status}</small>
              </div>
              <dl>
                <Metric label="Routing" value={run.routingAccuracy} />
                <Metric label="Context" value={run.contextRelevance} />
                <Metric label="Faithfulness" value={run.faithfulness} />
                <Metric label="Latency" value={run.latency} />
              </dl>
              <button className="secondary-action" onClick={onOpenDetail}>Open RAGXplain detail</button>
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}

function EvaluationDetailScreen() {
  return (
    <section className="page-stack">
      <PanelHeader eyebrow="Evaluation Detail" title="RAGXplain" />
      <div className="trace-board">
        {["User query", "Complexity classifier", "Route decision", "Retriever", "Post-retriever", "Prompt assembly", "LLM response", "Citation validator"].map((step, index) => (
          <article key={step}>
            <span>{index + 1}</span>
            <strong>{step}</strong>
            <p>{step === "Route decision" ? "Adaptive route selects L2 or L3/L4 based on classifier output." : "Captured in the trace metadata for each answer."}</p>
          </article>
        ))}
      </div>
    </section>
  );
}

function AnalyticsScreen() {
  const [tab, setTab] = useState("tokens");
  return (
    <section className="page-stack">
      <PanelHeader eyebrow="Usage Analytics" title="Analytics" />
      <div className="tabs">
        <button className={tab === "tokens" ? "active" : ""} onClick={() => setTab("tokens")}>Token statistics</button>
        <button className={tab === "feedback" ? "active" : ""} onClick={() => setTab("feedback")}>Detailed Statistics & Feedbacks</button>
      </div>
      {tab === "tokens" ? (
        <div className="metrics-grid">
          {tokenStats.map((stat) => (
            <article className="metric-card" key={stat.label}>
              <small>{stat.label}</small>
              <strong>{stat.value}</strong>
              <span>{stat.delta}</span>
            </article>
          ))}
        </div>
      ) : (
        <div className="table-panel">
          <table>
            <thead>
              <tr>
                <th>User</th>
                <th>Rating</th>
                <th>Topic</th>
                <th>Feedback</th>
              </tr>
            </thead>
            <tbody>
              {feedbackRows.map((row) => (
                <tr key={`${row.user}-${row.topic}`}>
                  <td>{row.user}</td>
                  <td>{row.rating}</td>
                  <td>{row.topic}</td>
                  <td>{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function PanelHeader({ eyebrow, title }) {
  return (
    <header className="panel-header">
      <p className="eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
    </header>
  );
}

function SelectField({ label, value, options, onChange }) {
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatDateTime(value) {
  if (!value) return "Not indexed";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatMetadataValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (Array.isArray(value)) return value.length ? value.map(formatMetadataValue).join(", ") : "[]";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function knowledgeConfigurationFromRecord(knowledgeBase) {
  return sanitizeKnowledgeConfiguration(knowledgeBase?.metadata?.configuration || defaultKnowledgeConfiguration);
}

function sanitizeKnowledgeConfiguration(configuration) {
  const raw = { ...defaultKnowledgeConfiguration, ...(configuration || {}) };
  const chunkSize = clampNumber(raw.chunk_size, 800, 100, 12000);
  const strategy = chunkingStrategies.some((item) => item.value === raw.chunking_strategy)
    ? raw.chunking_strategy
    : defaultKnowledgeConfiguration.chunking_strategy;
  const provider = String(raw.embedding_provider || "").toLowerCase() === supportedEmbeddingProvider.toLowerCase()
    ? supportedEmbeddingProvider
    : defaultKnowledgeConfiguration.embedding_provider;
  const models = supportedLocalEmbeddingModels;
  const embeddingModel = models.includes(raw.embedding_model) ? raw.embedding_model : models[0];
  return {
    chunking_strategy: strategy,
    chunk_size: chunkSize,
    chunk_overlap: chunkingStrategyUsesOverlap(strategy)
      ? clampNumber(raw.chunk_overlap, 120, 0, Math.max(chunkSize - 1, 0))
      : 0,
    embedding_provider: provider,
    embedding_model: embeddingModel
  };
}

function chunkingStrategyUsesOverlap(strategy) {
  return ["sliding_window_overlap", "semantic", "recursive", "hierarchical_parent_child"].includes(strategy);
}

function clampNumber(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(Math.trunc(parsed), max));
}

function createId() {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function hasSeenSplash() {
  try {
    return window.sessionStorage.getItem(SPLASH_SEEN_KEY) === "true";
  } catch {
    return false;
  }
}

function markSplashSeen() {
  try {
    window.sessionStorage.setItem(SPLASH_SEEN_KEY, "true");
  } catch {
    // Storage can be unavailable in hardened browser modes; falling through is safe.
  }
}
