import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BookOpen,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  CircleUserRound,
  ClipboardList,
  Copy,
  Cpu,
  Database,
  Eye,
  EyeOff,
  ExternalLink,
  FileText,
  Filter,
  GitBranch,
  Globe,
  HardDrive,
  Home,
  Layers,
  LogIn,
  LogOut,
  MessageSquarePlus,
  Paperclip,
  Pencil,
  Pin,
  PinOff,
  Plus,
  RefreshCw,
  RotateCw,
  Save,
  Search,
  SendHorizontal,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  UserPlus,
  X
} from "lucide-react";
import {
  askQuestion,
  askQuestionStream,
  createChatConfiguration,
  createEvaluationRun,
  createKnowledgeBase,
  createModelConnection,
  createModelDeploymentFromTemplate,
  deleteChatConfiguration,
  deleteChatConversation,
  deleteEvaluationRun,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  deleteModelDeployment,
  getModelUsageSummary,
  getCurrentUser,
  getKnowledgeProcessingTrace,
  getRagxplainViewerUrl,
  ingestWebsiteSource,
  listChatConfigurations,
  listChatConversations,
  listChatMessages,
  listEvaluationCases,
  listEvaluationRuns,
  listKnowledgeChunks,
  listKnowledgeDocuments,
  listKnowledgeBases,
  listKnowledgeIndexVersions,
  listModelDeployments,
  listModelConnections,
  listModelProviders,
  listModelUsage,
  clearAuthToken,
  hasAuthToken,
  login as loginUser,
  reindexKnowledgeBase,
  submitFeedback,
  testModelDeployment,
  testModelDeploymentDraft,
  testModelConnection,
  listConnectionModels,
  signup as signupUser,
  updateChatConfiguration,
  updateChatConversation,
  updateKnowledgeBase,
  updateModelDeployment,
  updateModelConnection,
  uploadKnowledgeSource
} from "./api.js";
import {
  architectureLayers,
  feedbackRows,
  tokenStats
} from "./data.js";

const navItems = [
  { id: "main", label: "Main", icon: Home },
  { id: "knowledge", label: "Knowledge Bases", icon: Database },
  { id: "model-farm", label: "AI Models", icon: Cpu },
  { id: "evaluation", label: "Evaluation", icon: ClipboardList },
  { id: "analytics", label: "Analytics", icon: BarChart3 }
];

const routes = [
  { value: "Adaptive", label: "Adaptive" },
  { value: "L1 Direct", label: "L1 Direct Generation" },
  { value: "L2 Simple RAG", label: "L2 Simple RAG" },
  { value: "L3 Complex RAG", label: "L3 Complex RAG" }
];
const responseStructures = [
  "Concise answer with bullets and cited workflow context",
  "Step-by-step workflow guidance",
  "Executive summary then details",
  "Detailed answer with assumptions and risks",
  "JSON-style structured response"
];
const chatbotTones = ["Professional", "Friendly", "Formal", "Technical", "Coaching"];
const modelCapabilities = ["generation", "embedding", "rerank", "judge", "planner", "classifier"];
const defaultGeneratorDeploymentId = "model-local-extractive";
const defaultEmbeddingDeploymentId = "model-local-hash-384";
const COMPOSER_MAX_CHARACTERS = 10000;
const CONFIGURATION_DISPLAY_ID_LENGTH = 12;
const CONFIGURATION_DISPLAY_ID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
const AI_LOGOS_PATH = "/AdaptiveRAG/img/logo_ai_models/";
const defaultChatConfigurationDraft = {
  chatConfigurationId: "",
  configurationCode: createConfigurationCode(),
  configurationCreatedAt: "",
  configurationUpdatedAt: "",
  configurationName: "Balanced workflow assistant",
  configurationDescription: "Default configuration for concise business workflow answers.",
  welcomeMessage: "Welcome to Adaptive RAG Studio. Ask a workflow question and I will route it through the selected classifier, retriever, and generator path.",
  conversationStarters: [
    "What are the main steps in this workflow?",
    "Summarize the selected knowledge base.",
    "Which documents explain this process best?"
  ],
  generatorDeploymentId: defaultGeneratorDeploymentId,
  fallbackDeploymentIds: [],
  rerankerDeploymentId: "",
  plannerDeploymentId: "",
  generationParameters: { temperature: 0.2, max_tokens: 500 },
  citationsEnabled: true,
  generatorProvider: "Local",
  generatorModel: "extractive",
  responseStructure: "Concise answer with bullets and cited workflow context",
  tone: "Professional",
  humorLevel: 0,
  systemPrompt: "You are an Adaptive RAG assistant for business workflow question answering. Answer using retrieved workflow context when available.",
  predefinedPrompt: "Answer clearly, mention uncertainty, and cite relevant workflow evidence when retrieval is used."
};
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
  embedding_deployment_id: defaultEmbeddingDeploymentId,
  external_processing_allowed: false,
  embedding_provider: "Local",
  embedding_model: "hash-embedding-384"
};
const SPLASH_SEEN_KEY = "aragbiz:splash-seen";
const SPLASH_DURATION_MS = 1900;
const MAIN_LAYOUT_STORAGE_KEY = "aragbiz:main-layout";
const MAIN_LAYOUT_LIMITS = {
  history: { min: 220, max: 360 },
  config: { min: 320, max: 520 },
  chatMin: 520,
  collapsed: 56
};
const DEFAULT_MAIN_LAYOUT = {
  historyWidth: 250,
  configWidth: 420,
  historyCollapsed: false,
  configCollapsed: false
};

export default function App() {
  const [screen, setScreen] = useState(() => (hasSeenSplash() ? "login" : "splash"));
  const [signedIn, setSignedIn] = useState(false);
  const [authReady, setAuthReady] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState("");
  const [selectedEvaluationDetail, setSelectedEvaluationDetail] = useState(null);
  const [confirmation, setConfirmation] = useState(null);
  const confirmationResolverRef = useRef(null);

  useEffect(() => {
    let active = true;
    if (!hasAuthToken()) {
      setAuthReady(true);
      return () => {
        active = false;
      };
    }
    getCurrentUser()
      .then((user) => {
        if (!active) return;
        setCurrentUser(user);
        setSignedIn(true);
        if (hasSeenSplash()) setScreen("main");
      })
      .catch(() => {
        clearAuthToken();
      })
      .finally(() => {
        if (active) setAuthReady(true);
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (screen !== "splash" || !authReady) return undefined;
    const timer = window.setTimeout(() => {
      markSplashSeen();
      setScreen(signedIn ? "main" : "login");
    }, SPLASH_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, [authReady, screen, signedIn]);

  async function enterStudio(mode, payload) {
    const result = mode === "signup" ? await signupUser(payload) : await loginUser(payload);
    setCurrentUser(result.user);
    setSignedIn(true);
    setScreen("main");
  }

  function leaveStudio() {
    clearAuthToken();
    setCurrentUser(null);
    setSignedIn(false);
    setScreen("login");
  }

  function confirmAction(options = {}) {
    return new Promise((resolve) => {
      confirmationResolverRef.current = resolve;
      setConfirmation({
        title: "Confirm action",
        message: "Are you sure you want to continue?",
        confirmLabel: "Confirm",
        cancelLabel: "Cancel",
        tone: "danger",
        ...options
      });
    });
  }

  function resolveConfirmation(confirmed) {
    if (confirmationResolverRef.current) {
      confirmationResolverRef.current(confirmed);
      confirmationResolverRef.current = null;
    }
    setConfirmation(null);
  }

  if (!authReady || screen === "splash") {
    return <Splash />;
  }

  if (!signedIn && screen === "login") {
    return <AuthScreen mode="login" onSubmit={(payload) => enterStudio("login", payload)} onSwitch={() => setScreen("signup")} />;
  }

  if (!signedIn && screen === "signup") {
    return <AuthScreen mode="signup" onSubmit={(payload) => enterStudio("signup", payload)} onSwitch={() => setScreen("login")} />;
  }

  return (
    <Shell activeScreen={screen} onNavigate={setScreen} user={currentUser} onSignOut={leaveStudio}>
      {screen === "main" && (
        <MainScreen
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
          confirmAction={confirmAction}
        />
      )}
      {screen === "knowledge" && (
        <KnowledgeBasesScreen
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
          confirmAction={confirmAction}
        />
      )}
      {screen === "model-farm" && <AIModelsScreen confirmAction={confirmAction} />}
      {screen === "evaluation" && (
        <EvaluationScreen
          selectedKnowledgeBaseId={selectedKnowledgeBaseId}
          onSelectKnowledgeBase={setSelectedKnowledgeBaseId}
          confirmAction={confirmAction}
          onOpenDetail={(run, evaluationCase, view = "case") => {
            setSelectedEvaluationDetail({ run, evaluationCase, view });
            setScreen("evaluation-detail");
          }}
        />
      )}
      {screen === "evaluation-detail" && (
        <EvaluationDetailScreen
          detail={selectedEvaluationDetail}
          onBack={() => setScreen("evaluation")}
        />
      )}
      {screen === "analytics" && <AnalyticsScreen />}
      <ConfirmationDialog
        confirmation={confirmation}
        onCancel={() => resolveConfirmation(false)}
        onConfirm={() => resolveConfirmation(true)}
      />
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
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  if (isSignup) {
    return <SignupScreen onSubmit={onSubmit} onSwitch={onSwitch} />;
  }

  async function submitLogin(event) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      await onSubmit({ email: email.trim(), password });
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-layout login-screen">
      <form className="auth-card login-form" onSubmit={submitLogin}>
        <div className="auth-logo">
          <span className="logo-dot" />
          <strong className="eyebrow">Adaptive RAG Studio</strong>
        </div>
        <h1>Login</h1>
        <label>
          Email
          <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" autoComplete="email" required />
        </label>
        <label>
          Password
          <input value={password} onChange={(event) => setPassword(event.target.value)} type="password" autoComplete="current-password" required />
        </label>
        {error && <p className="auth-error" role="alert">{error}</p>}
        <button className="primary-action" type="submit" disabled={submitting}>
          <IconLabel icon={LogIn} size={20}>{submitting ? "Signing in..." : "Login"}</IconLabel>
        </button>
        <button className="text-action" type="button" onClick={onSwitch}>
          <IconLabel icon={UserPlus}>Need an account? Sign up</IconLabel>
        </button>
      </form>
    </main>
  );
}

function SignupScreen({ onSubmit, onSwitch }) {
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    captcha: false,
    consent: false
  });
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
  }

  async function submitSignup(event) {
    event.preventDefault();
    if (!form.captcha || !form.consent) {
      setError("Confirm the verification and consent fields before creating an account.");
      return;
    }
    setError("");
    setSubmitting(true);
    try {
      await onSubmit({
        first_name: form.firstName.trim(),
        last_name: form.lastName.trim(),
        email: form.email.trim(),
        password: form.password
      });
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-layout signup-layout login-screen">
      <form className="signup-card login-form" onSubmit={submitSignup}>
        <div className="auth-logo">
          <span className="logo-dot" />
          <strong>Business Worklow Question Answering</strong>
        </div>
        <h1>Sign up new account</h1>
        <div className="signup-row">
          <label>
            First name
            <input value={form.firstName} onChange={(event) => updateField("firstName", event.target.value)} type="text" autoComplete="given-name" />
          </label>
          <label>
            Last name
            <input value={form.lastName} onChange={(event) => updateField("lastName", event.target.value)} type="text" autoComplete="family-name" />
          </label>
        </div>
        <label>
          Email
          <input value={form.email} onChange={(event) => updateField("email", event.target.value)} type="email" autoComplete="email" required />
        </label>
        <label>
          Password
          <input value={form.password} onChange={(event) => updateField("password", event.target.value)} type="password" autoComplete="new-password" minLength={8} required />
        </label>
        <div className="recaptcha-box" aria-label="reCAPTCHA verification placeholder">
          <label className="captcha-check">
            <input checked={form.captcha} onChange={(event) => updateField("captcha", event.target.checked)} type="checkbox" />
            <span>I'm not a robot</span>
          </label>
          <div className="captcha-brand">
            <span><IconOnly icon={RefreshCw} size={16} /></span>
            <small>reCAPTCHA</small>
          </div>
        </div>
        <label className="signup-consent">
          <input checked={form.consent} onChange={(event) => updateField("consent", event.target.checked)} type="checkbox" />
          <span>
            I agree to use Adaptive RAG Studio as an AI-powered system. I will verify answers since AI can make mistakes.
          </span>
        </label>
        {error && <p className="auth-error" role="alert">{error}</p>}
        <button className="primary-action signup-submit" type="submit" disabled={submitting}>
          <IconLabel icon={UserPlus} size={20}>{submitting ? "Creating account..." : "Agree and start"}</IconLabel>
        </button>
        <button className="text-action signup-login-link" type="button" onClick={onSwitch}>
          <IconLabel icon={LogIn}>Already have an account? Log in</IconLabel>
        </button>
      </form>
    </main>
  );
}

function Shell({ activeScreen, onNavigate, user, onSignOut, children }) {
  const userLabel = user?.first_name || user?.email || "User";
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo-dot" />
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
              <IconLabel icon={item.icon} size={18}>{item.label}</IconLabel>
            </button>
          ))}
        </nav>
        <div className="topbar-actions">
          <span className="user-chip" title={user?.email || ""}>
            <IconLabel icon={CircleUserRound}>{userLabel}</IconLabel>
          </span>
          <button type="button" aria-label="Sign out" title="Sign out" onClick={onSignOut}>
            <IconOnly icon={LogOut} size={18} />
          </button>
        </div>
      </header>
      <main className="workspace">{children}</main>
    </div>
  );
}

function ConfirmationDialog({ confirmation, onCancel, onConfirm }) {
  useEffect(() => {
    if (!confirmation) return undefined;
    function handleKeyDown(event) {
      if (event.key === "Escape") onCancel();
      if (event.key === "Enter") onConfirm();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [confirmation, onCancel, onConfirm]);

  if (!confirmation) return null;
  const isDanger = confirmation.tone !== "neutral";
  return (
    <div className="confirmation-backdrop" role="presentation" onMouseDown={onCancel}>
      <section
        className={`confirmation-dialog ${isDanger ? "danger" : "neutral"}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="confirmation-title"
        aria-describedby="confirmation-message"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="confirmation-icon" aria-hidden="true">
          <IconOnly icon={isDanger ? AlertTriangle : CheckCircle2} size={22} />
        </div>
        <div className="confirmation-content">
          <h2 id="confirmation-title">{confirmation.title}</h2>
          <p id="confirmation-message">{confirmation.message}</p>
          {confirmation.detail && <p className="confirmation-detail">{confirmation.detail}</p>}
        </div>
        <div className="confirmation-actions">
          <button className="secondary-action" type="button" onClick={onCancel}>
            {confirmation.cancelLabel || "Cancel"}
          </button>
          <button className={`primary-action ${isDanger ? "danger-confirm" : ""}`} type="button" onClick={onConfirm}>
            {confirmation.confirmLabel || "Confirm"}
          </button>
        </div>
      </section>
    </div>
  );
}

function MainScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase, confirmAction }) {
  const mainGridRef = useRef(null);
  const [messages, setMessages] = useState(() => welcomeMessagesFromConfig(defaultChatConfigurationDraft));
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [popup, setPopup] = useState(null);
  const [knowledgeBaseOptions, setKnowledgeBaseOptions] = useState([]);
  const [selectedFilterDocumentIds, setSelectedFilterDocumentIds] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [historyQuery, setHistoryQuery] = useState("");
  const [recentConversations, setRecentConversations] = useState([]);
  const [libraryConversations, setLibraryConversations] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [chatConfigurations, setChatConfigurations] = useState([]);
  const [modelDeployments, setModelDeployments] = useState([]);
  const [configurationStatus, setConfigurationStatus] = useState("");
  const [layout, setLayout] = useState(loadMainLayout);
  const [config, setConfig] = useState({
    classifier: "DistilBERT",
    classifierDeploymentId: "",
    queryEmbeddingDeploymentId: "",
    route: "Adaptive",
    retrievalMode: "Hybrid",
    topK: 6,
    reranker: true,
    citations: true,
    ...defaultChatConfigurationDraft
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

  useEffect(() => {
    refreshChatConfigurations();
    refreshMainModelDeployments();
  }, []);

  useEffect(() => {
    if (activeConversationId) return;
    setMessages((current) => (
      current.length === 0 || current.every(isWelcomeMessage)
        ? welcomeMessagesFromConfig(config)
        : current
    ));
  }, [activeConversationId, config.welcomeMessage]);

  useEffect(() => {
    setSelectedFilterDocumentIds([]);
  }, [selectedKnowledgeBaseId]);

  async function refreshMainModelDeployments() {
    try {
      const [generators, embeddings, classifiers, rerankers, planners] = await Promise.all([
        listModelDeployments({ capability: "generation", enabled: true }),
        listModelDeployments({ capability: "embedding", enabled: true }),
        listModelDeployments({ capability: "classifier", enabled: true }),
        listModelDeployments({ capability: "rerank", enabled: true }),
        listModelDeployments({ capability: "planner", enabled: true })
      ]);
      const byId = new Map();
      [...generators, ...embeddings, ...classifiers, ...rerankers, ...planners].forEach((deployment) => byId.set(deployment.id, deployment));
      setModelDeployments(Array.from(byId.values()));
    } catch (error) {
      setConfigurationStatus(`AI Models unavailable: ${error.message}`);
      setModelDeployments([]);
    }
  }

  useEffect(() => {
    saveMainLayout(layout);
  }, [layout]);

  useEffect(() => {
    if (!feedbackStatus) return undefined;
    const timer = window.setTimeout(() => setFeedbackStatus(""), 4200);
    return () => window.clearTimeout(timer);
  }, [feedbackStatus]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshConversationLists(historyQuery);
    }, 220);
    return () => window.clearTimeout(timer);
  }, [historyQuery]);

  async function refreshConversationLists(query = historyQuery) {
    setIsHistoryLoading(true);
    try {
      const [recents, library] = await Promise.all([
        listChatConversations({ query, section: "recents" }),
        listChatConversations({ query, section: "library" })
      ]);
      setRecentConversations(recents);
      setLibraryConversations(library);
    } catch (error) {
      setFeedbackStatus(`Chat history unavailable: ${error.message}`);
      setRecentConversations([]);
      setLibraryConversations([]);
    } finally {
      setIsHistoryLoading(false);
    }
  }

  async function refreshChatConfigurations(preferredId = config.chatConfigurationId) {
    try {
      const items = await listChatConfigurations();
      setChatConfigurations(items);
      const selected = preferredId ? items.find((item) => item.id === preferredId) : null;
      if (selected) {
        setConfig((current) => applyChatConfigurationToDraft(current, selected));
      }
      return items;
    } catch (error) {
      setConfigurationStatus(`Configuration presets unavailable: ${error.message}`);
      setChatConfigurations([]);
      return [];
    }
  }

  function selectChatConfiguration(configurationId) {
    const selected = chatConfigurations.find((item) => item.id === configurationId);
    if (!selected) {
      setConfig((current) => ({
        ...current,
        chatConfigurationId: "",
        configurationCode: createConfigurationCode(existingConfigurationCodes(chatConfigurations)),
        configurationCreatedAt: "",
        configurationUpdatedAt: "",
        configurationName: defaultChatConfigurationDraft.configurationName,
        configurationDescription: defaultChatConfigurationDraft.configurationDescription
      }));
      setConfigurationStatus("Using unsaved draft configuration");
      return;
    }
    setConfig((current) => applyChatConfigurationToDraft(current, selected));
    setConfigurationStatus(`Loaded "${selected.name}"`);
  }

  async function saveChatConfigurationAsNew() {
    try {
      const created = await createChatConfiguration(chatConfigurationPayloadFromDraft({
        ...config,
        chatConfigurationId: "",
        configurationCode: createConfigurationCode(existingConfigurationCodes(chatConfigurations))
      }));
      await refreshChatConfigurations(created.id);
      setConfig((current) => applyChatConfigurationToDraft(current, created));
      setConfigurationStatus(`Saved "${created.name}"`);
    } catch (error) {
      setConfigurationStatus(`Save configuration failed: ${error.message}`);
    }
  }

  async function updateSelectedChatConfiguration() {
    if (!config.chatConfigurationId) {
      await saveChatConfigurationAsNew();
      return;
    }
    try {
      const updated = await updateChatConfiguration(config.chatConfigurationId, chatConfigurationPayloadFromDraft(config));
      await refreshChatConfigurations(updated.id);
      setConfig((current) => applyChatConfigurationToDraft(current, updated));
      setConfigurationStatus(`Updated "${updated.name}"`);
    } catch (error) {
      setConfigurationStatus(`Update configuration failed: ${error.message}`);
    }
  }

  async function deleteSelectedChatConfiguration() {
    if (!config.chatConfigurationId) {
      setConfigurationStatus("Select a saved configuration before deleting.");
      return;
    }
    const confirmed = await confirmAction({
      title: "Delete configuration",
      message: `Delete "${config.configurationName}" (${config.configurationCode})? Existing conversations keep their saved snapshots, but this preset will no longer be selectable.`,
      confirmLabel: "Delete",
      tone: "danger"
    });
    if (!confirmed) return;
    try {
      await deleteChatConfiguration(config.chatConfigurationId);
      const items = await refreshChatConfigurations("");
      setConfig((current) => ({
        ...current,
        chatConfigurationId: "",
        configurationCode: createConfigurationCode(existingConfigurationCodes(items)),
        configurationCreatedAt: "",
        configurationUpdatedAt: "",
        configurationName: defaultChatConfigurationDraft.configurationName,
        configurationDescription: defaultChatConfigurationDraft.configurationDescription
      }));
      setConfigurationStatus(`Deleted "${config.configurationName}"`);
    } catch (error) {
      setConfigurationStatus(`Delete configuration failed: ${error.message}`);
    }
  }

  async function startNewChat() {
    setActiveConversationId("");
    setMessages(welcomeMessagesFromConfig(config));
    setQuestion("");
    setPopup(null);
  }

  async function selectConversation(conversation) {
    setActiveConversationId(conversation.id);
    if (conversation.knowledge_base_id) onSelectKnowledgeBase(conversation.knowledge_base_id);
    setConfig((current) => {
      let next = {
        ...current,
        route: routeLabelFromMode(conversation.route_mode),
        retrievalMode: retrievalModeLabel(conversation.retrieval_mode),
        topK: conversation.top_k || current.topK
      };
      const savedConfiguration = chatConfigurations.find((item) => item.id === conversation.chat_configuration_id);
      if (savedConfiguration) {
        next = applyChatConfigurationToDraft(next, savedConfiguration);
      } else if (conversation.metadata?.chat_configuration) {
        next = applyChatConfigurationSnapshotToDraft(next, conversation.metadata.chat_configuration, conversation.chat_configuration_id || "");
      }
      return next;
    });
    try {
      const records = await listChatMessages(conversation.id);
      setMessages(messagesFromChatRecords(records));
    } catch (error) {
      setFeedbackStatus(`Conversation load failed: ${error.message}`);
    }
  }

  async function togglePinnedConversation(conversation) {
    try {
      await updateChatConversation(conversation.id, { pinned: !conversation.pinned });
      await refreshConversationLists();
    } catch (error) {
      setFeedbackStatus(`Pin update failed: ${error.message}`);
    }
  }

  async function renameConversation(conversation, nextTitle) {
    const cleanedTitle = String(nextTitle || "").trim();
    if (!cleanedTitle || cleanedTitle === conversation.title) return true;
    try {
      await updateChatConversation(conversation.id, { title: cleanedTitle });
      await refreshConversationLists();
      return true;
    } catch (error) {
      setFeedbackStatus(`Rename chat failed: ${error.message}`);
      return false;
    }
  }

  async function removeConversation(conversation) {
    const confirmed = await confirmAction({
      title: "Delete chat?",
      message: `Delete "${conversation.title || "this chat"}"?`,
      detail: "This removes the saved conversation and its messages from chat history.",
      confirmLabel: "Delete chat"
    });
    if (!confirmed) return;
    try {
      await deleteChatConversation(conversation.id);
      if (activeConversationId === conversation.id) await startNewChat();
      await refreshConversationLists();
    } catch (error) {
      setFeedbackStatus(`Delete chat failed: ${error.message}`);
    }
  }

  async function sendQuestion() {
    const trimmed = question.trim();
    if (!trimmed) return;
    const mode = answerModeFromRoute(config.route);
    const requiresKnowledgeBase = mode !== "direct";
    if (requiresKnowledgeBase && !selectedKnowledgeBaseId) {
      setFeedbackStatus("Select a knowledge base before using Adaptive, L2 Simple RAG, or L3 Complex RAG.");
      return;
    }
    if (!isValidChatConfigurationDraft(config)) {
      setFeedbackStatus("Select or save a chatbot configuration before sending.");
      return;
    }
    const selectedKnowledgeConfiguration = selectedKnowledgeBase
      ? knowledgeConfigurationFromRecord(selectedKnowledgeBase)
      : {};
    const configuredDeploymentIds = [
      config.generatorDeploymentId,
      ...(config.fallbackDeploymentIds || []),
      config.plannerDeploymentId,
      config.queryEmbeddingDeploymentId,
      config.classifierDeploymentId,
      config.reranker ? config.rerankerDeploymentId : ""
    ].filter(Boolean);
    const remoteDeployment = configuredDeploymentIds
      .map((deploymentId) => modelDeployments.find((deployment) => deployment.id === deploymentId))
      .find((deployment) => deployment && deployment.locality !== "local");
    if (
      selectedKnowledgeBase
      && remoteDeployment
      && !selectedKnowledgeConfiguration.external_processing_allowed
    ) {
      setFeedbackStatus(
        `Remote model "${remoteDeployment.name}" is blocked for this knowledge base. `
        + "Open Knowledge Bases, modify the selected knowledge base, and enable Allow remote model processing."
      );
      return;
    }
    const assistantMessageId = createId();
    const streamRequestId = createId();
    let persistedConversationId = activeConversationId || "";
    let persistedAssistantMessageId = "";
    let streamPollTimer = null;
    let streamFinished = false;
    const userMessage = { id: createId(), role: "user", content: trimmed };
    const assistantPlaceholder = {
      id: assistantMessageId,
      question: trimmed,
      role: "assistant",
      content: "",
      contexts: [],
      metadata: { complexity_label: "pending", trace_steps: [] },
      status: "streaming",
      streaming: true,
      streamingStatus: "Connecting to Adaptive RAG..."
    };
    setMessages((current) => [...current.filter((message) => !isWelcomeMessage(message)), userMessage, assistantPlaceholder]);
    setQuestion("");
    setIsLoading(true);
    const stopStreamPolling = () => {
      streamFinished = true;
      if (streamPollTimer) {
        window.clearInterval(streamPollTimer);
        streamPollTimer = null;
      }
    };
    const patchStreamingAssistant = (patchOrUpdater) => {
      setMessages((current) => current.map((message) => {
        const metadata = message.metadata || {};
        const isTarget = message.id === assistantMessageId
          || (persistedAssistantMessageId && message.id === persistedAssistantMessageId)
          || metadata.request_id === streamRequestId
          || (persistedAssistantMessageId && metadata.assistant_message_id === persistedAssistantMessageId);
        if (!isTarget) return message;
        const patch = typeof patchOrUpdater === "function" ? patchOrUpdater(message) : patchOrUpdater;
        const { id: _ignoredId, ...safePatch } = patch || {};
        return { ...message, ...safePatch };
      }));
    };
    const startStreamPolling = (conversationId = "", assistantId = "", requestId = "") => {
      if ((!conversationId && !requestId) || streamPollTimer) return;
      streamPollTimer = window.setInterval(async () => {
        if (streamFinished) return;
        try {
          let records = [];
          let resolvedConversationId = conversationId || persistedConversationId;
          if (resolvedConversationId) {
            records = await listChatMessages(resolvedConversationId);
          } else {
            const conversations = await listChatConversations({ section: "recents" });
            for (const conversation of conversations.slice(0, 6)) {
              const candidateRecords = await listChatMessages(conversation.id);
              const candidate = candidateRecords.find((record) => (
                record.role === "assistant"
                && (record.request_id === requestId || record.metadata?.request_id === requestId)
              ));
              if (candidate) {
                records = candidateRecords;
                resolvedConversationId = conversation.id;
                persistedConversationId = conversation.id;
                setActiveConversationId(conversation.id);
                break;
              }
            }
          }
          const assistantRecord = records.find((record) => assistantId && record.id === assistantId)
            || records.find((record) => record.role === "assistant" && (record.request_id === requestId || record.metadata?.request_id === requestId))
            || [...records].reverse().find((record) => record.role === "assistant" && record.request_id);
          if (!assistantRecord) return;
          persistedAssistantMessageId = assistantRecord.id || persistedAssistantMessageId;
          patchStreamingAssistant({
            question: assistantRecord.metadata?.question || trimmed,
            role: "assistant",
            content: assistantRecord.content || "",
            contexts: assistantRecord.contexts || [],
            metadata: {
              ...(assistantRecord.metadata || {}),
              request_id: assistantRecord.request_id || assistantRecord.metadata?.request_id || requestId || streamRequestId,
              assistant_message_id: assistantRecord.id || assistantRecord.metadata?.assistant_message_id
            },
            status: assistantRecord.status || "streaming",
            streaming: ["pending", "streaming"].includes(assistantRecord.status || ""),
            streamingStatus: ["pending", "streaming"].includes(assistantRecord.status || "")
              ? "Streaming answer..."
              : ""
          });
          persistedConversationId = resolvedConversationId || persistedConversationId;
          if (["completed", "failed", "cancelled"].includes(assistantRecord.status || "")) {
            stopStreamPolling();
            setIsLoading(false);
            await refreshConversationLists();
          }
        } catch {
          // Polling is a fallback only; the primary SSE stream may still complete.
        }
      }, 1800);
    };
    try {
      const response = await askQuestionStream(trimmed, {
        requestId: streamRequestId,
        conversationId: activeConversationId,
        knowledgeBaseId: selectedKnowledgeBaseId,
        mode,
        retrievalMode: retrievalModeValue(config.retrievalMode),
        topK: config.topK,
        documentIds: selectedFilterDocumentIds,
        chatConfigurationId: config.chatConfigurationId || null,
        chatConfiguration: chatConfigurationPayloadFromDraft(config)
      }, (event) => {
        if (event.type === "started") {
          const serverAssistantId = event.data?.assistant_message_id;
          const serverConversationId = event.data?.conversation_id;
          const serverRequestId = event.data?.request_id || streamRequestId;
          if (serverConversationId) {
            persistedConversationId = serverConversationId;
            setActiveConversationId(serverConversationId);
          }
          persistedAssistantMessageId = serverAssistantId || persistedAssistantMessageId;
          patchStreamingAssistant((message) => ({
            metadata: {
              ...(message.metadata || {}),
              request_id: serverRequestId,
              user_message_id: event.data?.user_message_id || message.metadata?.user_message_id,
              assistant_message_id: serverAssistantId || message.metadata?.assistant_message_id
            },
            streamingStatus: serverAssistantId ? "Route is running..." : "Request accepted. Preparing route..."
          }));
          startStreamPolling(serverConversationId || persistedConversationId, serverAssistantId || persistedAssistantMessageId, serverRequestId);
        }
        if (event.type === "trace") {
          patchStreamingAssistant((message) => ({
            metadata: {
              ...(message.metadata || {}),
              trace_steps: [...(message.metadata?.trace_steps || []), event.data]
            },
            streamingStatus: event.data?.detail || event.data?.step || "Running Adaptive RAG..."
          }));
        }
        if (event.type === "sources") {
          patchStreamingAssistant({
            contexts: event.data?.contexts || [],
            streamingStatus: "Sources retrieved. Generating answer..."
          });
        }
        if (event.type === "delta") {
          patchStreamingAssistant((message) => ({
            content: `${message.content || ""}${event.data?.text || ""}`,
            streamingStatus: "Streaming answer..."
          }));
        }
      });
      if (response.conversation_id && response.conversation_id !== activeConversationId) {
        setActiveConversationId(response.conversation_id);
      }
      stopStreamPolling();
      patchStreamingAssistant({
          question: trimmed,
          role: "assistant",
          content: response.answer,
          contexts: response.contexts,
        metadata: {
          ...(response.metadata || {}),
          request_id: response.metadata?.request_id || streamRequestId,
          assistant_message_id: response.metadata?.assistant_message_id || persistedAssistantMessageId
        },
        status: "completed",
        streaming: false,
        streamingStatus: ""
      });
      await refreshConversationLists();
    } catch (error) {
      stopStreamPolling();
      patchStreamingAssistant({
          question: trimmed,
          role: "assistant",
          content: `Answer request failed: ${error.message}`,
          contexts: [],
        metadata: {
          error: error.message,
          complexity_label: "unknown",
          trace_steps: [],
          request_id: streamRequestId,
          assistant_message_id: persistedAssistantMessageId
        },
        status: "failed",
        streaming: false,
        streamingStatus: ""
      });
    } finally {
      stopStreamPolling();
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

  function togglePanel(panel) {
    setLayout((current) => ({
      ...current,
      [`${panel}Collapsed`]: !current[`${panel}Collapsed`]
    }));
  }

  function beginPanelResize(panel, event) {
    if ((panel === "history" && layout.historyCollapsed) || (panel === "config" && layout.configCollapsed)) return;
    const grid = mainGridRef.current;
    if (!grid || window.matchMedia("(max-width: 1180px)").matches) return;
    event.preventDefault();
    const bounds = grid.getBoundingClientRect();
    const startX = event.clientX;
    const startLayout = { ...layout };
    const historyBase = startLayout.historyCollapsed ? MAIN_LAYOUT_LIMITS.collapsed : startLayout.historyWidth;
    const configBase = startLayout.configCollapsed ? MAIN_LAYOUT_LIMITS.collapsed : startLayout.configWidth;
    const maxHistory = Math.max(
      MAIN_LAYOUT_LIMITS.history.min,
      Math.min(MAIN_LAYOUT_LIMITS.history.max, bounds.width - configBase - MAIN_LAYOUT_LIMITS.chatMin - 16)
    );
    const maxConfig = Math.max(
      MAIN_LAYOUT_LIMITS.config.min,
      Math.min(MAIN_LAYOUT_LIMITS.config.max, bounds.width - historyBase - MAIN_LAYOUT_LIMITS.chatMin - 16)
    );

    function handleMove(moveEvent) {
      const delta = moveEvent.clientX - startX;
      setLayout((current) => {
        if (panel === "history") {
          return {
            ...current,
            historyWidth: clampNumber(startLayout.historyWidth + delta, startLayout.historyWidth, MAIN_LAYOUT_LIMITS.history.min, maxHistory)
          };
        }
        return {
          ...current,
          configWidth: clampNumber(startLayout.configWidth - delta, startLayout.configWidth, MAIN_LAYOUT_LIMITS.config.min, maxConfig)
        };
      });
    }

    function handleUp() {
      document.body.classList.remove("is-resizing-panels");
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    }

    document.body.classList.add("is-resizing-panels");
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
  }

  const selectedKnowledgeBase = knowledgeBaseOptions.find((item) => item.id === selectedKnowledgeBaseId);
  const historyWidth = layout.historyCollapsed ? MAIN_LAYOUT_LIMITS.collapsed : layout.historyWidth;
  const configWidth = layout.configCollapsed ? MAIN_LAYOUT_LIMITS.collapsed : layout.configWidth;

  return (
    <section
      ref={mainGridRef}
      className={`main-grid ${layout.historyCollapsed ? "history-collapsed" : ""} ${layout.configCollapsed ? "config-collapsed" : ""}`}
      style={{ "--history-width": `${historyWidth}px`, "--config-width": `${configWidth}px` }}
    >
      <ConversationHistory
        collapsed={layout.historyCollapsed}
        onToggle={() => togglePanel("history")}
        onNewChat={startNewChat}
        searchQuery={historyQuery}
        onSearchChange={setHistoryQuery}
        libraryConversations={libraryConversations}
        recentConversations={recentConversations}
        activeConversationId={activeConversationId}
        isLoading={isHistoryLoading}
        onSelectConversation={selectConversation}
        onTogglePinned={togglePinnedConversation}
        onRenameConversation={renameConversation}
        onDeleteConversation={removeConversation}
      />
      <PanelResizeHandle side="left" label="Resize chat history" onPointerDown={(event) => beginPanelResize("history", event)} />
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
        selectedFilterDocumentIds={selectedFilterDocumentIds}
        onFilterDocumentsChange={setSelectedFilterDocumentIds}
        welcomeMessage={config.welcomeMessage}
        conversationStarters={config.conversationStarters}
        selectedRoute={config.route}
        requiresKnowledgeBase={answerModeFromRoute(config.route) !== "direct"}
      />
      <PanelResizeHandle side="right" label="Resize configuration" onPointerDown={(event) => beginPanelResize("config", event)} />
      <RagConfiguration
        config={config}
        setConfig={setConfig}
        selectedKnowledgeBase={selectedKnowledgeBase}
        chatConfigurations={chatConfigurations}
        modelDeployments={modelDeployments}
        onRefreshModelDeployments={refreshMainModelDeployments}
        configurationStatus={configurationStatus}
        onSelectChatConfiguration={selectChatConfiguration}
        onSaveConfiguration={saveChatConfigurationAsNew}
        onUpdateConfiguration={updateSelectedChatConfiguration}
        onDeleteConfiguration={deleteSelectedChatConfiguration}
        collapsed={layout.configCollapsed}
        onToggle={() => togglePanel("config")}
      />
      {feedbackStatus && (
        <div className="toast">
          <span>{feedbackStatus}</span>
          <button type="button" onClick={() => setFeedbackStatus("")} aria-label="Dismiss notification">
            <IconOnly icon={X} size={14} />
          </button>
        </div>
      )}
      {popup && <TraceModal popup={popup} onClose={() => setPopup(null)} />}
    </section>
  );
}

function PanelResizeHandle({ side, label, onPointerDown }) {
  return <div className={`panel-resize-handle ${side}`} role="separator" aria-label={label} onPointerDown={onPointerDown} />;
}
function ConversationHistory({
  collapsed,
  onToggle,
  onNewChat,
  searchQuery,
  onSearchChange,
  libraryConversations,
  recentConversations,
  activeConversationId,
  isLoading,
  onSelectConversation,
  onTogglePinned,
  onRenameConversation,
  onDeleteConversation
}) {
  if (collapsed) {
    return (
      <aside className="history-panel panel-rail history-rail">
        <button className="panel-rail-button" type="button" onClick={onToggle} aria-label="Expand chat history">
          <span>History</span>
        </button>
      </aside>
    );
  }
  return (
    <aside className="history-panel">
      <header className="panel-titlebar">
        <div>
          <p className="eyebrow">Chat history</p>
          <h2>Conversations</h2>
        </div>
        <button className="panel-collapse-button" type="button" onClick={onToggle} aria-label="Collapse chat history"><IconOnly icon={ChevronLeft} /></button>
      </header>
      <button className="new-chat-action" type="button" onClick={onNewChat}><IconLabel icon={MessageSquarePlus}>New chat</IconLabel></button>
      <label className="history-search">
        <IconOnly icon={Search} size={16} />
        <input
          value={searchQuery}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder="Search chats"
          aria-label="Search chats"
        />
      </label>
      <ConversationSection
        title="Library"
        icon={BookOpen}
        conversations={libraryConversations}
        emptyText={searchQuery ? "No pinned chats match." : "Pinned chats appear here."}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
        onTogglePinned={onTogglePinned}
        onRenameConversation={onRenameConversation}
        onDeleteConversation={onDeleteConversation}
      />
      <ConversationSection
        title="Recents"
        icon={RotateCw}
        conversations={recentConversations}
        emptyText={isLoading ? "Loading chats..." : searchQuery ? "No recent chats match." : "Start a new chat to build history."}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
        onTogglePinned={onTogglePinned}
        onRenameConversation={onRenameConversation}
        onDeleteConversation={onDeleteConversation}
      />
    </aside>
  );
}

function ConversationSection({
  title,
  icon,
  conversations,
  emptyText,
  activeConversationId,
  onSelectConversation,
  onTogglePinned,
  onRenameConversation,
  onDeleteConversation
}) {
  const [editingConversationId, setEditingConversationId] = useState("");
  const [editingTitle, setEditingTitle] = useState("");

  function beginRename(conversation) {
    setEditingConversationId(conversation.id);
    setEditingTitle(conversation.title || "New chat");
  }

  function cancelRename() {
    setEditingConversationId("");
    setEditingTitle("");
  }

  async function saveRename(conversation) {
    const nextTitle = editingTitle.trim();
    if (!nextTitle || nextTitle === conversation.title) {
      cancelRename();
      return;
    }
    const saved = await onRenameConversation(conversation, nextTitle);
    if (saved !== false) cancelRename();
  }

  return (
    <section className="conversation-section">
      <h3><IconLabel icon={icon}>{title}</IconLabel></h3>
      <div className="conversation-list">
        {conversations.length === 0 ? (
          <p className="history-empty">{emptyText}</p>
        ) : conversations.map((conversation) => {
          const isEditing = editingConversationId === conversation.id;
          return (
            <article
              key={conversation.id}
              className={`conversation-item ${conversation.id === activeConversationId ? "active" : ""} ${isEditing ? "editing" : ""}`}
            >
              {isEditing ? (
                <form
                  className="conversation-title-edit"
                  onSubmit={(event) => {
                    event.preventDefault();
                    saveRename(conversation);
                  }}
                >
                  <input
                    value={editingTitle}
                    autoFocus
                    aria-label="Chat name"
                    onChange={(event) => setEditingTitle(event.target.value)}
                    onBlur={() => saveRename(conversation)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") {
                        event.preventDefault();
                        cancelRename();
                      }
                    }}
                  />
                </form>
              ) : (
                <button className="conversation-main" type="button" onClick={() => onSelectConversation(conversation)}>
                  <span className="conversation-title-block">
                    <strong title={conversation.title}>{chatHistoryDisplayTitle(conversation.title)}</strong>
                    <small>{conversation.route_mode || "adaptive"} - {conversation.retrieval_mode || "hybrid"}</small>
                  </span>
                </button>
              )}
              <div className="conversation-side">
                <em className="conversation-time">{formatShortDate(conversation.updated_at)}</em>
                <div className="conversation-actions">
                  <button
                    type="button"
                    aria-label={conversation.pinned ? "Unpin chat" : "Pin chat"}
                    onClick={() => onTogglePinned(conversation)}
                  >
                    <IconOnly icon={conversation.pinned ? PinOff : Pin} size={14} />
                  </button>
                  <button type="button" aria-label="Rename chat" onClick={() => beginRename(conversation)}>
                    <IconOnly icon={Pencil} size={14} />
                  </button>
                  <button type="button" aria-label="Delete chat" onClick={() => onDeleteConversation(conversation)}>
                    <IconOnly icon={Trash2} size={14} />
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
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
  onSelectKnowledgeBase,
  selectedFilterDocumentIds = [],
  onFilterDocumentsChange = () => {},
  welcomeMessage = "",
  conversationStarters = [],
  selectedRoute,
  requiresKnowledgeBase
}) {
  const messageListRef = useRef(null);
  const messageEndRef = useRef(null);
  const [isFilterOpen, setIsFilterOpen] = useState(false);
  const [filterDocuments, setFilterDocuments] = useState([]);
  const [isFilterLoading, setIsFilterLoading] = useState(false);
  const [filterError, setFilterError] = useState("");
  const [filterQuery, setFilterQuery] = useState("");
  const [draftFilterDocumentIds, setDraftFilterDocumentIds] = useState(selectedFilterDocumentIds);
  const selectedKnowledgeBase = knowledgeBases.find((knowledgeBase) => knowledgeBase.id === selectedKnowledgeBaseId);
  const chatPlaceholder = selectedKnowledgeBase
    ? `Send a message to "${selectedKnowledgeBase.name}"`
    : "Please select a knowledge base to start the conversation.";
  const selectedFilterCount = selectedFilterDocumentIds.length;
  const currentCharacterCount = question.length;
  const remainingCharacterCount = Math.max(COMPOSER_MAX_CHARACTERS - currentCharacterCount, 0);
  const characterUsagePercent = Math.min(100, Math.round((currentCharacterCount / COMPOSER_MAX_CHARACTERS) * 100));
  const shouldShowStarters = messages.length === 0 || messages.every(isWelcomeMessage);
  const visibleStarters = normalizeConversationStarters(conversationStarters);
  const scrollSignature = messages
    .map((message) => `${message.id}:${message.content?.length || 0}:${message.streamingStatus || ""}:${message.status || ""}`)
    .join("|");

  useEffect(() => {
    if (!isFilterOpen) setDraftFilterDocumentIds(selectedFilterDocumentIds);
  }, [selectedFilterDocumentIds, isFilterOpen]);

  useEffect(() => {
    setIsFilterOpen(false);
    setFilterDocuments([]);
    setFilterQuery("");
    setDraftFilterDocumentIds([]);
  }, [selectedKnowledgeBaseId]);

  useEffect(() => {
    const list = messageListRef.current;
    const end = messageEndRef.current;
    if (!list || !end) return;
    window.requestAnimationFrame(() => {
      end.scrollIntoView({ block: "end", behavior: "smooth" });
    });
  }, [scrollSignature]);

  async function openDocumentFilter() {
    if (!selectedKnowledgeBaseId) return;
    setIsFilterOpen(true);
    setFilterQuery("");
    setFilterError("");
    setDraftFilterDocumentIds(selectedFilterDocumentIds);
    setIsFilterLoading(true);
    try {
      const documents = await listKnowledgeDocuments(selectedKnowledgeBaseId);
      setFilterDocuments(documents);
    } catch (error) {
      setFilterError(`Document filter load failed: ${error.message}`);
      setFilterDocuments([]);
    } finally {
      setIsFilterLoading(false);
    }
  }

  function toggleDraftFilterDocument(documentId) {
    setDraftFilterDocumentIds((current) => (
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId]
    ));
  }

  function applyDocumentFilter() {
    onFilterDocumentsChange(draftFilterDocumentIds);
    setIsFilterOpen(false);
  }

  function clearDocumentFilter() {
    setDraftFilterDocumentIds([]);
    onFilterDocumentsChange([]);
    setIsFilterOpen(false);
  }

  const filteredDocuments = filterDocuments.filter((document) => documentMatchesFilterQuery(document, filterQuery));

  return (
    <section className="chat-panel">
      <header className="chat-titlebar">
        <div className="brain-mark"><BrainCircuit size={18} aria-hidden="true" /></div>
        <div>
          <h1>Business Workflow Question Answering</h1>
          <p>Business Workflow Question Answering AI Chatbot</p>
        </div>
      </header>
      <div className="message-list" ref={messageListRef}>
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            {message.role === "assistant" && <span className="avatar"><Bot size={16} aria-hidden="true" /></span>}
            <div className="message-body">
              {message.content ? (
                message.role === "assistant" ? (
                  <AssistantMessageContent content={message.content} />
                ) : (
                  <p className="user-message-content">{message.content}</p>
                )
              ) : (
                <p className="streaming-placeholder">{message.streamingStatus || "Preparing answer..."}</p>
              )}
              {message.streaming && (
                <div className="streaming-status">
                  <span className="streaming-dot" />
                  <span>{message.streamingStatus || "Streaming..."}</span>
                </div>
              )}
              {!message.streaming && ["failed", "cancelled", "pending", "streaming"].includes(message.status) && (
                <div className={`message-state message-state-${message.status}`}>
                  {message.status === "failed" && "Answer failed during streaming."}
                  {message.status === "cancelled" && "Answer stream was cancelled."}
                  {message.status === "pending" && "Answer is pending."}
                  {message.status === "streaming" && "Answer was still streaming when this chat was loaded."}
                </div>
              )}
            </div>
            {message.role === "assistant" && !message.streaming && !isWelcomeMessage(message) && (
              <div className="message-actions">
                <button onClick={() => onOpenPopup({ type: "source", message })}><IconLabel icon={Layers}>Sources</IconLabel></button>
                <button type="button"><IconLabel icon={Copy}>Copy</IconLabel></button>
                <button onClick={() => onOpenPopup({ type: "trace", message })}><IconLabel icon={GitBranch}>Trace</IconLabel></button>
                <button onClick={() => onFeedback(message, "up")}><IconLabel icon={ThumbsUp}>Useful</IconLabel></button>
                <button onClick={() => onFeedback(message, "down")}><IconLabel icon={ThumbsDown}>Needs work</IconLabel></button>
                <span><IconLabel icon={BrainCircuit}>{message.metadata?.complexity_label || "pending"}</IconLabel></span>
              </div>
            )}
          </article>
        ))}
        {shouldShowStarters && visibleStarters.length > 0 && (
          <div className="conversation-starters" aria-label="Conversation starters">
            {visibleStarters.map((starter) => (
              <button key={starter} type="button" onClick={() => setQuestion(starter)}>
                {starter}
              </button>
            ))}
          </div>
        )}
        <div ref={messageEndRef} className="message-list-end" aria-hidden="true" />
      </div>
      <div className="composer">
        <textarea
          aria-label="Chat message"
          placeholder={chatPlaceholder}
          maxLength={COMPOSER_MAX_CHARACTERS}
          value={question}
          onChange={(event) => setQuestion(event.target.value.slice(0, COMPOSER_MAX_CHARACTERS))}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              onSend();
            }
          }}
        />
        <div className="composer-tools">
          <span><IconLabel icon={Paperclip}>Attach</IconLabel></span>
          <button
            className={`composer-tool-button ${selectedFilterCount ? "active" : ""}`}
            type="button"
            onClick={openDocumentFilter}
            disabled={!selectedKnowledgeBaseId}
            title={selectedKnowledgeBaseId ? "Filter retrieval by selected documents" : "Select a knowledge base first"}
          >
            <IconLabel icon={Filter}>Filter{selectedFilterCount ? ` (${selectedFilterCount})` : ""}</IconLabel>
          </button>
          <span
            className={`char-usage-meter ${characterUsagePercent >= 90 ? "warning" : ""}`}
            tabIndex={0}
            aria-label={`Character usage ${characterUsagePercent} percent. Maximum ${formatInteger(COMPOSER_MAX_CHARACTERS)}, current ${formatInteger(currentCharacterCount)}, remaining ${formatInteger(remainingCharacterCount)}.`}
          >
            {characterUsagePercent}%
            <span className="char-usage-tooltip" role="tooltip">
              <strong>Character usage</strong>
              <span>Maximum: {formatInteger(COMPOSER_MAX_CHARACTERS)}</span>
              <span>Current: {formatInteger(currentCharacterCount)}</span>
              <span>Remaining: {formatInteger(remainingCharacterCount)}</span>
            </span>
          </span>
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
          <strong>{selectedRoute}</strong>
          <button className="send-action" onClick={onSend} disabled={isLoading || (requiresKnowledgeBase && !selectedKnowledgeBaseId)} aria-label="Send message"><SendHorizontal size={20} aria-hidden="true" /></button>
        </div>
      </div>
      {isFilterOpen && (
        <div className="document-filter-backdrop" role="presentation" onMouseDown={() => setIsFilterOpen(false)}>
          <section
            className="document-filter-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="document-filter-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div>
                <p className="eyebrow">Retrieval scope</p>
                <h2 id="document-filter-title">Filter documents</h2>
                <p>Search only inside selected documents from {selectedKnowledgeBase?.name || "the current Knowledge Base"}.</p>
              </div>
              <button className="icon-button" type="button" onClick={() => setIsFilterOpen(false)} aria-label="Close document filter">
                <IconOnly icon={X} size={18} />
              </button>
            </header>
            <label className="document-filter-search">
              <IconOnly icon={Search} size={16} />
              <input
                value={filterQuery}
                onChange={(event) => setFilterQuery(event.target.value)}
                placeholder="Search documents"
                aria-label="Search documents"
              />
            </label>
            <div className="document-filter-status">
              <span>{draftFilterDocumentIds.length ? `${draftFilterDocumentIds.length} selected` : "All documents included"}</span>
              <button
                className="text-action"
                type="button"
                onClick={() => setDraftFilterDocumentIds(filterDocuments.map((document) => document.id))}
                disabled={!filterDocuments.length}
              >
                Select all
              </button>
            </div>
            <div className="document-filter-list">
              {isFilterLoading && <p className="muted-text">Loading documents...</p>}
              {filterError && <p className="config-warning-note">{filterError}</p>}
              {!isFilterLoading && !filterError && filteredDocuments.length === 0 && (
                <p className="muted-text">{filterDocuments.length ? "No documents match your search." : "No documents found in this Knowledge Base."}</p>
              )}
              {!isFilterLoading && !filterError && filteredDocuments.map((document) => (
                <label key={document.id} className="document-filter-item">
                  <input
                    type="checkbox"
                    checked={draftFilterDocumentIds.includes(document.id)}
                    onChange={() => toggleDraftFilterDocument(document.id)}
                  />
                  <div>
                    <strong>{document.title || document.metadata?.filename || document.id}</strong>
                    <span>{document.metadata?.source_type || document.metadata?.mime_type || "Document"} · {document.text?.length || 0} chars</span>
                    <p>{compactPreview(document.text || document.metadata?.uri || document.id, 130)}</p>
                  </div>
                </label>
              ))}
            </div>
            <footer>
              <button className="secondary-action" type="button" onClick={clearDocumentFilter}>Clear filter</button>
              <button className="secondary-action" type="button" onClick={() => setIsFilterOpen(false)}>Cancel</button>
              <button className="primary-action" type="button" onClick={applyDocumentFilter}>
                Apply {draftFilterDocumentIds.length ? `${draftFilterDocumentIds.length} document${draftFilterDocumentIds.length === 1 ? "" : "s"}` : "all documents"}
              </button>
            </footer>
          </section>
        </div>
      )}
      <p className="ai-disclaimer">This is an AI-powered system. Please verify answers since AI can make mistakes.</p>
    </section>
  );
}

function RagConfiguration({
  config,
  setConfig,
  selectedKnowledgeBase,
  chatConfigurations = [],
  modelDeployments = [],
  onRefreshModelDeployments = () => {},
  configurationStatus = "",
  onSelectChatConfiguration = () => {},
  onSaveConfiguration = () => {},
  onUpdateConfiguration = () => {},
  onDeleteConfiguration = () => {},
  collapsed,
  onToggle
}) {
  const generatorDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("generation"));
  const embeddingDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("embedding"));
  const classifierDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("classifier"));
  const rerankerDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("rerank"));
  const plannerDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("planner"));
  const activeKnowledgeConfiguration = selectedKnowledgeBase ? knowledgeConfigurationFromRecord(selectedKnowledgeBase) : {};
  const activeEmbeddingDeploymentId = selectedKnowledgeBase ? activeKnowledgeConfiguration.embedding_deployment_id || "" : "";
  const executedQueryEmbeddingModel = selectedKnowledgeBase
    ? selectedKnowledgeBase.embedding_model || activeKnowledgeConfiguration.embedding_model || "Not indexed"
    : "";
  const selectedQueryEmbeddingDeploymentId = config.queryEmbeddingDeploymentId || activeEmbeddingDeploymentId || "";
  const selectedQueryEmbeddingDeployment = embeddingDeployments.find((deployment) => deployment.id === selectedQueryEmbeddingDeploymentId);
  const selectedQueryEmbeddingLabel = selectedQueryEmbeddingDeployment?.model
    || selectedQueryEmbeddingDeployment?.name
    || selectedQueryEmbeddingDeploymentId
    || executedQueryEmbeddingModel;
  const hasQueryEmbeddingOverride = Boolean(
    config.queryEmbeddingDeploymentId
    && activeEmbeddingDeploymentId
    && config.queryEmbeddingDeploymentId !== activeEmbeddingDeploymentId
  );
  const selectedChatConfiguration = chatConfigurations.find((item) => item.id === config.chatConfigurationId);
  const configurationCreatedAt = selectedChatConfiguration?.created_at || config.configurationCreatedAt || "";
  const configurationUpdatedAt = selectedChatConfiguration?.updated_at || config.configurationUpdatedAt || "";
  const [collapsedSections, setCollapsedSections] = useState({});
  function toggleCustomizerSection(sectionId) {
    setCollapsedSections((current) => ({ ...current, [sectionId]: !current[sectionId] }));
  }
  const queryEmbeddingOptions = selectedKnowledgeBase
    ? [
        {
          value: activeEmbeddingDeploymentId || "",
          label: `${executedQueryEmbeddingModel || "Active KB embedding"} (active KB model)`
        },
        ...embeddingDeployments
          .filter((deployment) => deployment.id !== activeEmbeddingDeploymentId)
          .map((deployment) => deploymentOption(deployment))
      ]
    : [{ value: "", label: "Select a knowledge base first", disabled: true }];
  if (collapsed) {
    return (
      <aside className="config-panel panel-rail config-rail">
        <button className="panel-rail-button" type="button" onClick={onToggle} aria-label="Expand RAG Customizer">
          <span>RAG</span>
        </button>
      </aside>
    );
  }
  return (
    <aside className="config-panel">
      <header className="panel-titlebar">
        <div>
          <p className="eyebrow">Runtime Configuration</p>
          <h2>RAG Customizer</h2>
        </div>
        <button className="panel-collapse-button" type="button" onClick={onToggle} aria-label="Collapse RAG Customizer"><IconOnly icon={ChevronRight} /></button>
      </header>
      <KnowledgeBaseSummary
        selectedKnowledgeBase={selectedKnowledgeBase}
        collapsed={Boolean(collapsedSections.knowledge)}
        onToggle={() => toggleCustomizerSection("knowledge")}
      />
      <div className="rag-customizer-scroll">
      <CollapsibleCustomizerSection
        title="General"
        description="Save or modify Runtime settings for General, Adaptive RAG, and Generator target & prompts."
        collapsed={Boolean(collapsedSections.general)}
        onToggle={() => toggleCustomizerSection("general")}
      >
        <SelectField
          label="Select configuration"
          value={config.chatConfigurationId}
          options={[
            { value: "", label: "New configuration" },
            ...chatConfigurations.map((item) => chatConfigurationOption(item))
          ]}
          onChange={onSelectChatConfiguration}
        />
        <dl className="configuration-meta-row">
          <div>
            <dt>Created</dt>
            <dd>{configurationCreatedAt ? formatDateTime(configurationCreatedAt) : "Not saved yet"}</dd>
          </div>
          <div>
            <dt>Last modified</dt>
            <dd>{configurationUpdatedAt ? formatDateTime(configurationUpdatedAt) : "Not saved yet"}</dd>
          </div>
        </dl>
        <label>
          Configuration ID
          <input
            value={config.configurationCode || ""}
            readOnly
            aria-readonly="true"
            title="Auto-generated ID. This value is not editable."
          />
        </label>
        <label>
          Configuration name
          <input
            value={config.configurationName}
            onChange={(event) => setConfig({ ...config, configurationName: event.target.value })}
            placeholder="Balanced workflow assistant"
          />
        </label>
        <label>
          Description
          <input
            value={config.configurationDescription}
            onChange={(event) => setConfig({ ...config, configurationDescription: event.target.value })}
            placeholder="Short purpose for this runtime configuration"
          />
        </label>
        <label>
          Welcome message
          <textarea
            value={config.welcomeMessage}
            onChange={(event) => setConfig({ ...config, welcomeMessage: event.target.value })}
            placeholder="Welcome users and explain what this assistant can do."
          />
        </label>
        <label>
          Conversation starter
          <textarea
            value={normalizeConversationStarters(config.conversationStarters).join("\n")}
            onChange={(event) => setConfig({ ...config, conversationStarters: parseConversationStarters(event.target.value) })}
            placeholder={"What are the main steps in this workflow?\nSummarize the selected knowledge base."}
          />
          <small>Enter one starter per line. These appear as quick-start buttons in a new chat.</small>
        </label>
      </CollapsibleCustomizerSection>
      <CollapsibleCustomizerSection
        title="Adaptive RAG"
        description="Routing, retrieval, and optional planning controls for the next answer."
        actions={(
          <button
            className="icon-button section-refresh-button"
            type="button"
            onClick={onRefreshModelDeployments}
            title="Refresh AI model options"
            aria-label="Refresh AI model options"
          >
            <IconOnly icon={RefreshCw} size={16} />
          </button>
        )}
        collapsed={Boolean(collapsedSections.adaptive)}
        onToggle={() => toggleCustomizerSection("adaptive")}
      >
        <SelectField
          label="Route strategy"
          value={config.route}
          options={routes}
          onChange={(route) => setConfig({ ...config, route })}
        />
        <div className="config-two-column">
          <SelectField
            label="Classifier model"
            value={config.classifierDeploymentId || ""}
            options={[
              { value: "", label: "Built-in trained classifier" },
              ...classifierDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(classifierDeploymentId) => {
              const deployment = modelDeployments.find((item) => item.id === classifierDeploymentId);
              setConfig({
                ...config,
                classifierDeploymentId,
                classifier: deployment?.model || deployment?.name || "Built-in trained classifier"
              });
            }}
          />
          <SelectField
            label="Planner model"
            value={config.plannerDeploymentId || ""}
            options={[
              { value: "", label: "Deterministic L3 decomposition" },
              ...plannerDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(plannerDeploymentId) => setConfig({ ...config, plannerDeploymentId })}
          />
        </div>
        <SelectField
          label="Query embedding generation model"
          value={selectedQueryEmbeddingDeploymentId}
          options={queryEmbeddingOptions}
          onChange={(queryEmbeddingDeploymentId) => setConfig({
            ...config,
            queryEmbeddingDeploymentId: queryEmbeddingDeploymentId === activeEmbeddingDeploymentId ? "" : queryEmbeddingDeploymentId
          })}
        />
        {hasQueryEmbeddingOverride && (
          <p className="config-warning-note">
            Warning: selected query embedding model ({selectedQueryEmbeddingLabel}) is different from the active Knowledge Base embedding
            ({executedQueryEmbeddingModel}). Dense retrieval should use matching query/document embeddings to avoid invalid similarity scores.
          </p>
        )}
        <div className="config-two-column">
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
        </div>
        <label className="check-row">
          <input
            type="checkbox"
            checked={config.reranker}
            onChange={(event) => setConfig({ ...config, reranker: event.target.checked })}
          />
          Enable reranker
        </label>
        {config.reranker && (
          <SelectField
            label="Reranker model"
            value={config.rerankerDeploymentId || ""}
            options={[
              { value: "", label: "Built-in lexical reranker" },
              ...rerankerDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(rerankerDeploymentId) => setConfig({ ...config, rerankerDeploymentId })}
          />
        )}
        <label className="check-row">
          <input
            type="checkbox"
            checked={config.citations}
            onChange={(event) => setConfig({ ...config, citations: event.target.checked })}
          />
          Citation validator
        </label>
      </CollapsibleCustomizerSection>
      <CollapsibleCustomizerSection
        title="Generator target & prompts"
        description="Generator settings are sent with each answer request. External providers execute when enabled in AI Models."
        collapsed={Boolean(collapsedSections.generator)}
        onToggle={() => toggleCustomizerSection("generator")}
      >
        <div className="config-two-column">
          <SelectField
            label="Generator model"
            value={config.generatorDeploymentId || ""}
            options={[
              { value: "", label: "Select enabled deployment" },
              ...generatorDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(generatorDeploymentId) => {
              const deployment = modelDeployments.find((item) => item.id === generatorDeploymentId);
              setConfig({
                ...config,
                generatorDeploymentId,
                generatorProvider: deployment?.provider || config.generatorProvider,
                generatorModel: deployment?.model || config.generatorModel
              });
            }}
          />
          <SelectField
            label="Fallback deployment"
            value={config.fallbackDeploymentIds?.[0] || ""}
            options={[
              { value: "", label: "No fallback" },
              ...generatorDeployments
                .filter((deployment) => deployment.id !== config.generatorDeploymentId)
                .map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(fallbackDeploymentId) => setConfig({ ...config, fallbackDeploymentIds: fallbackDeploymentId ? [fallbackDeploymentId] : [] })}
          />
        </div>
        {generatorDeployments.length === 0 && (
          <p className="config-status-note">No enabled generator deployment found. Add or enable one in AI Models.</p>
        )}
        <SelectField
          label="Response structure"
          value={config.responseStructure}
          options={responseStructures}
          onChange={(responseStructure) => setConfig({ ...config, responseStructure })}
        />
        <div className="config-two-column">
          <SelectField
            label="Tone"
            value={config.tone}
            options={chatbotTones}
            onChange={(tone) => setConfig({ ...config, tone })}
          />
          <label>
            Humor level
            <input
              type="range"
              min="0"
              max="5"
              value={config.humorLevel}
              onChange={(event) => setConfig({ ...config, humorLevel: Number(event.target.value) })}
            />
            <strong>{config.humorLevel}/5</strong>
          </label>
        </div>
        <label>
          System prompt
          <textarea
            value={config.systemPrompt}
            onChange={(event) => setConfig({ ...config, systemPrompt: event.target.value })}
          />
        </label>
        <label>
          Predefined prompt instructions
          <textarea
            value={config.predefinedPrompt}
            onChange={(event) => setConfig({ ...config, predefinedPrompt: event.target.value })}
          />
        </label>
      </CollapsibleCustomizerSection>
      <footer className="rag-customizer-actions">
        {configurationStatus && <p className="config-status-note">{configurationStatus}</p>}
        <div className="config-actions-row">
          <button
            className="secondary-action danger-action"
            type="button"
            onClick={onDeleteConfiguration}
            disabled={!config.chatConfigurationId}
            title={config.chatConfigurationId ? "Delete selected configuration" : "Select a saved configuration to delete"}
          >
            <IconLabel icon={Trash2}>Delete</IconLabel>
          </button>
          <button className="secondary-action" type="button" onClick={onSaveConfiguration}><IconLabel icon={Plus}>Save as new</IconLabel></button>
          <button className="primary-action" type="button" onClick={onUpdateConfiguration}><IconLabel icon={Save}>Save</IconLabel></button>
        </div>
      </footer>
      </div>
    </aside>
  );
}

function CollapsibleCustomizerSection({
  title,
  description = "",
  actions = null,
  collapsed = false,
  onToggle = () => {},
  children
}) {
  return (
    <section className={`config-section runtime-section collapsible-config-section ${collapsed ? "is-collapsed" : ""}`}>
      <header>
        <div>
          <h3>{title}</h3>
          {description && <small>{description}</small>}
        </div>
        <div className="section-header-actions">
          {actions}
          <button
            className="section-collapse-button"
            type="button"
            onClick={onToggle}
            aria-label={`${collapsed ? "Expand" : "Collapse"} ${title}`}
            aria-expanded={!collapsed}
          >
            <IconOnly icon={ChevronRight} size={16} />
          </button>
        </div>
      </header>
      {!collapsed && <div className="collapsible-config-body">{children}</div>}
    </section>
  );
}

function KnowledgeBaseSummary({ selectedKnowledgeBase, collapsed = false, onToggle = () => {} }) {
  const configuration = selectedKnowledgeBase ? knowledgeConfigurationFromRecord(selectedKnowledgeBase) : {};
  const queryEmbedding = selectedKnowledgeBase?.embedding_model || configuration.embedding_model || "Select a Knowledge Base";
  return (
    <section className={`route-summary-card knowledge-summary-card collapsible-config-section ${collapsed ? "is-collapsed" : ""}`}>
      <header>
        <div>
          <p className="eyebrow">Knowledge base</p>
          <h3>{selectedKnowledgeBase?.name || "No Knowledge Base"}</h3>
        </div>
        <div className="section-header-actions">
          <span className={`status-pill ${selectedKnowledgeBase?.status === "ready" ? "status-completed" : ""}`}>
            {selectedKnowledgeBase?.status || "Undefined"}
          </span>
          <button
            className="section-collapse-button"
            type="button"
            onClick={onToggle}
            aria-label={`${collapsed ? "Expand" : "Collapse"} Knowledge base`}
            aria-expanded={!collapsed}
          >
            <IconOnly icon={ChevronRight} size={16} />
          </button>
        </div>
      </header>
      {!collapsed && (
        <dl className="route-summary-list">
          <div>
            <dt>Knowledge base</dt>
            <dd>{selectedKnowledgeBase ? selectedKnowledgeBase.name : "Not selected"}</dd>
          </div>
          <div>
            <dt>Status</dt>
            <dd>{selectedKnowledgeBase?.status || "-"}</dd>
          </div>
          <div className="route-summary-wide">
            <dt>Description</dt>
            <dd>{selectedKnowledgeBase?.description || "No description"}</dd>
          </div>
          <div>
            <dt>Documents / chunks</dt>
            <dd>{selectedKnowledgeBase ? `${selectedKnowledgeBase.document_count} / ${selectedKnowledgeBase.chunk_count}` : "-"}</dd>
          </div>
          <div>
            <dt>Query embedding</dt>
            <dd>{queryEmbedding}</dd>
          </div>
          <div>
            <dt>External processing</dt>
            <dd>{configuration.external_processing_allowed ? "Allowed" : "Blocked"}</dd>
          </div>
          <div>
            <dt>Last indexed</dt>
            <dd>{formatDateTime(selectedKnowledgeBase?.updated_at)}</dd>
          </div>
        </dl>
      )}
    </section>
  );
}
function TraceModal({ popup, onClose }) {
  const { type, message } = popup;
  const contexts = message.contexts || [];
  const traceSteps = Array.isArray(message.metadata?.trace_steps) ? message.metadata.trace_steps : [];
  const [sourceQuery, setSourceQuery] = useState("");
  const [selectedSourceId, setSelectedSourceId] = useState(contexts[0]?.id || "");
  const [traceQuery, setTraceQuery] = useState("");
  const [selectedTraceIndex, setSelectedTraceIndex] = useState(0);

  useEffect(() => {
    setSourceQuery("");
    setSelectedSourceId(contexts[0]?.id || "");
    setTraceQuery("");
    setSelectedTraceIndex(0);
  }, [message.id, type]);

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Clipboard can be unavailable outside secure contexts; ignore silently.
    }
  }

  const filteredContexts = contexts.filter((context) => {
    const haystack = [
      context.text,
      context.metadata?.title,
      context.metadata?.document_id,
      context.metadata?.embedding_model,
      context.metadata?.source_subquery,
      context.metadata?.retrieval_step,
      context.mode
    ].filter(Boolean).join(" ").toLowerCase();
    return haystack.includes(sourceQuery.trim().toLowerCase());
  });
  const selectedSource = filteredContexts.find((context) => context.id === selectedSourceId) || filteredContexts[0];
  const traceItems = traceSteps.map((step, index) => ({ step, index }));
  const filteredTraceItems = traceItems.filter(({ step }) => {
    const haystack = [step.step, step.status, step.detail, JSON.stringify(step.metadata || {})].join(" ").toLowerCase();
    return haystack.includes(traceQuery.trim().toLowerCase());
  });
  const selectedTraceItem = filteredTraceItems.find((item) => item.index === selectedTraceIndex) || filteredTraceItems[0];
  const selectedTrace = selectedTraceItem?.step;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className={`modal ${type === "source" ? "source-modal" : "trace-modal"}`} role="dialog" aria-modal="true" onClick={(event) => event.stopPropagation()}>
        <header className="modal-header">
          <div>
            <h2><IconLabel icon={type === "source" ? Layers : GitBranch} size={20}>{type === "source" ? "Source Documents" : "Trace report"}</IconLabel></h2>
            <p>{type === "source" ? "Review the retrieved chunks used to ground this answer." : "Inspect route, retrieval, and generation steps for this answer."}</p>
          </div>
          <button className="icon-button modal-close" aria-label="Close modal" onClick={onClose}><IconOnly icon={X} /></button>
        </header>
        {type === "source" ? (
          contexts.length === 0 ? (
            <div className="empty-state source-empty-state">
              <strong>No source chunks</strong>
              <p>{message.metadata?.route_label || "This route"} did not retrieve knowledge-base context.</p>
              <small>L1 Direct Generation does not retrieve knowledge-base context, so there are no source chunks for that route.</small>
            </div>
          ) : (
            <div className="source-modal-grid">
              <aside className="source-sidebar">
                <label className="source-search">
                  <span><IconLabel icon={Search}>Search sources</IconLabel></span>
                  <input value={sourceQuery} onChange={(event) => setSourceQuery(event.target.value)} placeholder="Filter by title, text, model..." />
                </label>
                <div className="source-count-row">
                  <span>{filteredContexts.length} of {contexts.length} chunks</span>
                  <button type="button" onClick={() => setSourceQuery("")}><IconLabel icon={X}>Clear</IconLabel></button>
                </div>
                {filteredContexts.map((context) => (
                  <button
                    key={context.id}
                    className={`source-result ${context.id === selectedSource?.id ? "selected" : ""}`}
                    type="button"
                    onClick={() => setSelectedSourceId(context.id)}
                  >
                    <span>{context.metadata?.source_type || "Chunk"}</span>
                    <strong>{context.metadata?.title || `Knowledge chunk ${context.rank}`}</strong>
                    <small>{context.mode || message.metadata?.retrieval_mode || "retrieval"} - chunk {context.metadata?.chunk_index ?? context.rank}</small>
                    {context.metadata?.source_subquery && <small>Step {context.metadata?.subquery_index}: {context.metadata.source_subquery}</small>}
                    <em>{Math.round(Number(context.score || 0) * 100)}% match</em>
                  </button>
                ))}
              </aside>
              <article className="source-preview">
                {selectedSource ? (
                  <>
                    <header className="source-preview-header">
                      <div>
                        <p className="eyebrow">Selected chunk</p>
                        <h3>{selectedSource.metadata?.title || "Knowledge-base source"}</h3>
                      </div>
                      <button className="secondary-action compact-action" type="button" onClick={() => copyText(selectedSource.text)}><IconLabel icon={Copy}>Copy text</IconLabel></button>
                    </header>
                    <div className="metadata-chip-row">
                      <span>Rank {selectedSource.rank}</span>
                      <span>{Math.round(Number(selectedSource.score || 0) * 100)}% match</span>
                      <span>{selectedSource.mode || message.metadata?.retrieval_mode || "retrieval"}</span>
                      <span>Chunk {selectedSource.metadata?.chunk_index ?? "-"}</span>
                      {selectedSource.metadata?.aggregated_rank != null && <span>Aggregated rank {selectedSource.metadata.aggregated_rank}</span>}
                      {selectedSource.metadata?.retrieval_step && <span>{selectedSource.metadata.retrieval_step}</span>}
                      <span>{selectedSource.metadata?.embedding_model || "No embedding model"}</span>
                    </div>
                    <dl className="source-facts">
                      <div><dt>Document ID</dt><dd>{selectedSource.metadata?.document_id || "-"}</dd></div>
                      <div><dt>Source ID</dt><dd>{selectedSource.metadata?.source_id || "-"}</dd></div>
                      <div><dt>Token count</dt><dd>{selectedSource.metadata?.token_count || selectedSource.metadata?.chunk_size || "-"}</dd></div>
                      <div><dt>Knowledge base</dt><dd>{selectedSource.metadata?.knowledge_base_id || message.metadata?.knowledge_base_name || "-"}</dd></div>
                      <div><dt>Source subquery</dt><dd>{selectedSource.metadata?.source_subquery || "-"}</dd></div>
                      <div><dt>Original rank</dt><dd>{selectedSource.metadata?.original_rank ?? "-"}</dd></div>
                      <div><dt>Subquery coverage</dt><dd>{selectedSource.metadata?.subquery_coverage ?? "-"}</dd></div>
                    </dl>
                    <div className="source-text-preview">{selectedSource.text}</div>
                  </>
                ) : (
                  <div className="empty-state"><strong>No matching chunks</strong><p>Try a broader source filter.</p></div>
                )}
              </article>
            </div>
          )
        ) : (
          <div className="trace-layout">
            <aside className="trace-steps">
              <h3><IconLabel icon={GitBranch}>Steps Overview</IconLabel></h3>
              <label className="trace-side-search">
                <input value={traceQuery} onChange={(event) => setTraceQuery(event.target.value)} placeholder="Search trace" />
              </label>
              {filteredTraceItems.length === 0 ? (
                <p className="muted-text">No trace steps match.</p>
              ) : filteredTraceItems.map(({ step, index }) => (
                <button key={`${step.step}-${index}`} className={index === selectedTraceItem?.index ? "active" : ""} type="button" onClick={() => setSelectedTraceIndex(index)}>
                  <span>{index + 1}. {step.step}</span>
                  <em>{step.status}</em>
                </button>
              ))}
            </aside>
            <div className="trace-content">
              <TraceSummary metadata={message.metadata || {}} />
              {selectedTrace ? (
                <article className="trace-step-detail">
                  <header>
                    <div>
                      <p className="eyebrow">Selected step</p>
                      <h3>{selectedTrace.step}</h3>
                    </div>
                    <span className="status-pill">{selectedTrace.status}</span>
                  </header>
                  <p>{selectedTrace.detail}</p>
                  <div className="trace-metadata-grid">
                    {Object.entries(selectedTrace.metadata || {}).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key}</dt>
                        <dd>{formatMetadataValue(value)}</dd>
                      </div>
                    ))}
                    {Object.keys(selectedTrace.metadata || {}).length === 0 && <p className="muted-text">No metadata for this step.</p>}
                  </div>
                  <details className="raw-json-panel">
                    <summary>Raw step JSON</summary>
                    <pre>{JSON.stringify(selectedTrace, null, 2)}</pre>
                  </details>
                </article>
              ) : (
                <div className="empty-state"><strong>No trace metadata returned</strong><p>Ask a new question to generate trace steps.</p></div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function TraceSummary({ metadata }) {
  return (
    <section className="trace-summary-card">
      <div><dt>Route</dt><dd>{metadata.route_label || metadata.route_level || "-"}</dd></div>
      <div><dt>Complexity</dt><dd>{metadata.complexity_label || "-"}</dd></div>
      <div><dt>Retrieval</dt><dd>{metadata.retrieval_mode || "none"}</dd></div>
      <div><dt>Top K</dt><dd>{metadata.top_k ?? "-"}</dd></div>
      <div><dt>Multi-step</dt><dd>{metadata.multi_step ? "Yes" : "No"}</dd></div>
      <div><dt>Subqueries</dt><dd>{metadata.decomposed_queries?.length || 0}</dd></div>
      <div><dt>Latency</dt><dd>{metadata.latency_ms ? `${metadata.latency_ms} ms` : "-"}</dd></div>
      <div><dt>Knowledge base</dt><dd>{metadata.knowledge_base_name || "-"}</dd></div>
    </section>
  );
}
function KnowledgeBasesScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase, confirmAction }) {
  const [items, setItems] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [chunks, setChunks] = useState([]);
  const [processingTrace, setProcessingTrace] = useState([]);
  const [indexVersions, setIndexVersions] = useState([]);
  const [embeddingDeployments, setEmbeddingDeployments] = useState([]);
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
      listModelDeployments({ capability: "embedding", enabled: true })
        .then(setEmbeddingDeployments)
        .catch(() => setEmbeddingDeployments([]));
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
      setIndexVersions([]);
      return;
    }
    setIsLoadingDocuments(true);
    try {
      const [nextDocuments, nextChunks, nextTrace, nextVersions] = await Promise.all([
        listKnowledgeDocuments(knowledgeBaseId),
        listKnowledgeChunks(knowledgeBaseId),
        getKnowledgeProcessingTrace(knowledgeBaseId),
        listKnowledgeIndexVersions(knowledgeBaseId).catch(() => [])
      ]);
      setDocuments(nextDocuments);
      setChunks(nextChunks);
      setProcessingTrace(nextTrace);
      setIndexVersions(nextVersions);
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
    const confirmed = await confirmAction({
      title: "Delete document?",
      message: `Delete "${document?.title || "this document"}"?`,
      detail: "All chunks, embeddings, and processing trace linked to this document will be removed.",
      confirmLabel: "Delete document"
    });
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
    const confirmed = await confirmAction({
      title: "Delete knowledge base?",
      message: `Delete "${selectedKnowledgeBase.name}" and all of its documents?`,
      detail: "This removes the knowledge base, documents, chunks, embeddings, and active index metadata.",
      confirmLabel: "Delete knowledge base"
    });
    if (!confirmed) return;
    setActionStatus("Deleting knowledge base and all documents...");
    try {
      await deleteKnowledgeBase(selectedKnowledgeBase.id);
      setActionStatus("Knowledge base deleted");
      onSelectKnowledgeBase("");
      setDocuments([]);
      setChunks([]);
      setProcessingTrace([]);
      setIndexVersions([]);
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
        <button className="primary-action" onClick={() => setKnowledgeBaseModal({ mode: "create" })}><IconLabel icon={Plus} size={20}>Add knowledge base</IconLabel></button>
        <button className="secondary-action" onClick={refreshKnowledgeBases}><IconLabel icon={RefreshCw}>Refresh</IconLabel></button>
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
              <button className="primary-action" onClick={() => setKnowledgeBaseModal({ mode: "edit", knowledgeBase: selectedKnowledgeBase })}>
                <IconLabel icon={Pencil}>Update knowledge base</IconLabel>
              </button>
              <button className="secondary-action" onClick={() => handleReindex(selectedKnowledgeBase.id)}>
                <IconLabel icon={RotateCw}>Re-index</IconLabel>
              </button>
              <button className="secondary-action danger-action" onClick={handleDeleteKnowledgeBase}>
                <IconLabel icon={Trash2}>Delete knowledge base</IconLabel>
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
                <IconLabel icon={FileText}>Documents</IconLabel>
              </button>
              <button
                className={detailTab === "indexes" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={detailTab === "indexes"}
                onClick={() => setDetailTab("indexes")}
              >
                <IconLabel icon={Layers}>Index versions</IconLabel>
              </button>
              <button
                className={detailTab === "trace" ? "active" : ""}
                type="button"
                role="tab"
                aria-selected={detailTab === "trace"}
                onClick={() => setDetailTab("trace")}
              >
                <IconLabel icon={GitBranch}>Processing trace</IconLabel>
              </button>
            </div>
            {detailTab === "documents" ? (
              <div className="document-list detail-tab-panel">
                {isLoadingDocuments && <p className="muted-text">Loading documents...</p>}
                {!isLoadingDocuments && documents.length === 0 && <p className="muted-text">No documents yet. Use Modify KB to add local files or website sources.</p>}
                {documents.map((document, index) => (
                  <details key={document.id} className="document-card document-accordion" open={index === 0}>
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
                        <IconLabel icon={Trash2}>Delete document</IconLabel>
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
            ) : detailTab === "indexes" ? (
              <IndexVersionPanel versions={indexVersions} />
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
          embeddingDeployments={embeddingDeployments}
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
                  <IconLabel icon={chunk.has_embedding ? CheckCircle2 : Activity}>{chunk.has_embedding ? "embedded" : "not embedded"}</IconLabel>
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

function IndexVersionPanel({ versions = [] }) {
  if (!versions.length) {
    return <div className="empty-state"><strong>No index versions yet</strong><p>Index versions appear after upload, website ingestion, or re-indexing.</p></div>;
  }
  return (
    <div className="table-panel compact-table detail-tab-panel">
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Embedding deployment</th>
            <th>Model</th>
            <th>Dimension</th>
            <th>Documents</th>
            <th>Chunks</th>
            <th>Created</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((version) => (
            <tr key={version.id}>
              <td><span className={`status-pill status-${version.status}`}>{version.status}</span></td>
              <td>{version.embedding_deployment_id || "-"}</td>
              <td>{version.embedding_model || "-"}</td>
              <td>{version.embedding_dimension || "-"}</td>
              <td>{version.document_count}</td>
              <td>{version.chunk_count}</td>
              <td>{formatDateTime(version.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
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
        <span><IconLabel icon={Activity}>{steps.length} steps</IconLabel></span>
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
  embeddingDeployments = [],
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
            <h2>{isEdit ? "Update knowledge base" : "Add knowledge base"}</h2>
            <p>{isEdit ? "Update name, description, add more documents, or delete existing documents." : "Enter details and select multiple files or a public website source."}</p>
          </div>
          <button className="icon-button modal-close" aria-label="Close modal" onClick={onClose}><IconOnly icon={X} /></button>
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
                Embedding deployment
                <select
                  value={configuration.embedding_deployment_id || ""}
                  onChange={(event) => {
                    const deployment = embeddingDeployments.find((item) => item.id === event.target.value);
                    setConfiguration((current) => ({
                      ...current,
                      embedding_deployment_id: event.target.value,
                      embedding_provider: deployment?.provider || current.embedding_provider,
                      embedding_model: deployment?.model || current.embedding_model
                    }));
                  }}
                >
                  <option value="">Select enabled embedding deployment</option>
                  {configuration.embedding_deployment_id && !embeddingDeployments.some((deployment) => deployment.id === configuration.embedding_deployment_id) && (
                    <option value={configuration.embedding_deployment_id}>{configuration.embedding_deployment_id} (current)</option>
                  )}
                  {embeddingDeployments.map((deployment) => (
                    <option key={deployment.id} value={deployment.id}>
                      {shortDeploymentLabel(deployment)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Resolved embedding model
                <input value={`${configuration.embedding_provider} / ${configuration.embedding_model}`} readOnly />
              </label>
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={Boolean(configuration.external_processing_allowed)}
                onChange={(event) => setConfiguration((current) => ({ ...current, external_processing_allowed: event.target.checked }))}
              />
              Allow remote model processing for this knowledge base
            </label>
            <p className="muted-text compact-muted">
              Embeddings come from the active AI Models deployment. Remote deployments require this knowledge base to explicitly allow external processing.
            </p>
          </section>
          <div className="source-selector">
            <button type="button" className={sourceType === "upload" ? "selected" : ""} onClick={() => setSourceType("upload")}>
              <IconLabel icon={HardDrive}>{isEdit ? "Add local files" : "Local device"}</IconLabel>
            </button>
            <button type="button" className={sourceType === "website" ? "selected" : ""} onClick={() => setSourceType("website")}>
              <IconLabel icon={Globe}>{isEdit ? "Add website" : "Public website"}</IconLabel>
            </button>
            <button type="button" disabled><IconLabel icon={Database}>Google Drive soon</IconLabel></button>
            <button type="button" disabled><IconLabel icon={Database}>OneDrive soon</IconLabel></button>
            <button type="button" disabled><IconLabel icon={Database}>Databases soon</IconLabel></button>
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
            <button type="button" className="secondary-action" onClick={onClose}><IconLabel icon={X}>Cancel</IconLabel></button>
            <button type="submit" className="primary-action" disabled={isSubmitting}>
              <IconLabel icon={isSubmitting ? RefreshCw : Save}>{isSubmitting ? "Processing..." : isEdit ? "Save knowledge base" : "Create knowledge base"}</IconLabel>
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}

function AIModelsScreen({ confirmAction }) {
  const [templates, setTemplates] = useState([]);
  const [connections, setConnections] = useState([]);
  const [deployments, setDeployments] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [usageSummary, setUsageSummary] = useState({});
  const [capabilityFilter, setCapabilityFilter] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [providerSearch, setProviderSearch] = useState("");
  const [providerTab, setProviderTab] = useState("provider");
  const [isAddOpen, setIsAddOpen] = useState(false);
  const [addStage, setAddStage] = useState("select");
  const [selectedTemplateId, setSelectedTemplateId] = useState("");
  const [wizard, setWizard] = useState(() => emptyModelFarmWizard());
  const [endpointDraft, setEndpointDraft] = useState(() => emptyEndpointDraft());
  const [editingDeployment, setEditingDeployment] = useState(null);
  const [createdDeploymentId, setCreatedDeploymentId] = useState("");
  const [showEndpointSecret, setShowEndpointSecret] = useState(false);
  const [endpointTestResult, setEndpointTestResult] = useState(null);
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState(false);

  useEffect(() => {
    refreshModelFarm();
  }, [capabilityFilter]);

  async function refreshModelFarm() {
    setIsLoading(true);
    try {
      const [nextTemplates, nextConnections, nextDeployments, nextSummary] = await Promise.all([
        listModelProviders(),
        listModelConnections(),
        listModelDeployments(),
        getModelUsageSummary()
      ]);
      setTemplates(nextTemplates);
      setConnections(nextConnections);
      setDeployments(capabilityFilter
        ? nextDeployments.filter((deployment) => deployment.capabilities?.includes(capabilityFilter))
        : nextDeployments
      );
      setUsageSummary(nextSummary || {});
      setStatus("");
    } catch (error) {
      setTemplates([]);
      setConnections([]);
      setDeployments([]);
      setStatus(`AI Models load failed: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  }

  function openAddModel() {
    setIsAddOpen(true);
    setAddStage("select");
    setProviderTab("provider");
    setProviderSearch("");
    setSelectedTemplateId("");
    setEditingDeployment(null);
    setCreatedDeploymentId("");
    setEndpointDraft(emptyEndpointDraft());
    setAvailableModels([]);
    setEndpointTestResult(null);
    setWizard(emptyModelFarmWizard());
    setStatus("");
  }

  function closeAddModel() {
    setIsAddOpen(false);
    setAddStage("select");
    setSelectedTemplateId("");
    setEditingDeployment(null);
    setCreatedDeploymentId("");
    setEndpointDraft(emptyEndpointDraft());
    setAvailableModels([]);
    setEndpointTestResult(null);
    setShowEndpointSecret(false);
  }

  function openEditModel(deployment) {
    const templateId = templateIdForDeployment(deployment, templates);
    setIsAddOpen(true);
    setAddStage("details");
    setProviderTab(deployment.locality === "local" ? "local" : "provider");
    setProviderSearch("");
    setSelectedTemplateId(templateId);
    setEditingDeployment(deployment);
    setCreatedDeploymentId(deployment.id);
    setEndpointDraft(endpointDraftFromDeployment(deployment));
    setAvailableModels([]);
    setEndpointTestResult(null);
    setWizard(modelFarmWizardFromDeployment(deployment));
    setShowEndpointSecret(false);
    setStatus("");
  }

  function applyTemplate(template) {
    if (!template) return;
    const defaults = template.deployment_defaults || {};
    setSelectedTemplateId(template.id);
    setCreatedDeploymentId("");
    setEndpointDraft({
      ...emptyEndpointDraft(defaults),
      name: uniqueConnectionName(`${template.provider_label || template.label} connection`, connections)
    });
    setAvailableModels([]);
    setEndpointTestResult(null);
    setWizard({
      ...emptyModelFarmWizard(),
      name: uniqueDeploymentName(defaults.name || template.provider_label || template.label || "AI model", deployments),
      model: defaults.model || template.model || "",
      capabilities: [...(defaults.capabilities || template.capabilities || [])],
      apiBase: defaults.api_base || "",
      credentialEnvRefs: cleanCredentialRefs(defaults.credential_env_refs || {}),
      defaultParameters: objectOrEmpty(defaults.default_parameters),
      limits: objectOrEmpty(defaults.limits),
      pricing: objectOrEmpty(defaults.pricing),
      monthlyBudgetUsd: Number(defaults.monthly_budget_usd || 0),
      hardBudget: defaults.hard_budget !== false,
      temperature: Number(defaults.default_parameters?.temperature ?? 0.2),
      maxTokens: Number(defaults.default_parameters?.max_tokens ?? defaults.limits?.max_output_tokens ?? 800),
      timeoutSeconds: Number(defaults.limits?.timeout_seconds || 60),
      dimension: Number(defaults.limits?.dimension || 0),
      inputPrice: Number(defaults.pricing?.input_per_million_tokens_usd || 0),
      outputPrice: Number(defaults.pricing?.output_per_million_tokens_usd || 0),
      metadata: objectOrEmpty(defaults.metadata)
    });
  }

  function updateEndpointDraft(patch) {
    setEndpointDraft((current) => ({ ...current, ...patch }));
    setEndpointTestResult(null);
  }

  function confirmProvider() {
    const template = selectedTemplate;
    if (!template) {
      setStatus("Select an AI model provider first.");
      return;
    }
    if (!template.creatable) {
      closeAddModel();
      setStatus(`${template.label} is already available as a local AI model.`);
      return;
    }
    setAddStage("details");
  }

  async function saveEndpointConnection() {
    const template = selectedTemplate;
    if (!template) throw new Error("Select an AI model provider first.");
    const apiBase = endpointDraft.url.trim() || template.deployment_defaults?.api_base || "";
    const apiKey = endpointDraft.apiKey.trim();
    if (requiresEndpointUrl(template) && !apiBase) throw new Error("Enter the endpoint URL.");
    if (!endpointDraft.connectionId && template.credential_fields?.includes("api_key") && !apiKey && !wizard.credentialEnvRefs?.api_key) {
      throw new Error("Enter an API key or configure its ARAGBIZ_MODEL_* environment variable.");
    }
    const provider = connectionProviderForTemplate(template);
    const connectionName = endpointDraft.name.trim()
      || uniqueConnectionName(`${template.provider_label || template.label} connection`, connections);
    const common = {
      name: connectionName,
      api_base: apiBase,
      ...(apiKey ? { credential_secrets: { api_key: apiKey } } : {})
    };
    const connection = endpointDraft.connectionId
      ? await updateModelConnection(endpointDraft.connectionId, common)
      : await createModelConnection({
          ...common,
          provider,
          access_path: template.access_path || (template.locality === "local" ? "local" : "production"),
          locality: template.locality || (["ollama", "vllm"].includes(provider) ? "local" : "remote"),
          credential_env_refs: cleanCredentialRefs(wizard.credentialEnvRefs || {}),
          enabled: false,
          metadata: { template_id: template.id }
        });
    setEndpointDraft((current) => ({
      ...current,
      connectionId: connection.id,
      name: connection.name,
      url: connection.api_base,
      apiKey: ""
    }));
    setConnections((current) => [connection, ...current.filter((item) => item.id !== connection.id)]);
    return connection;
  }

  async function createEndpointDeployment() {
    const template = selectedTemplate;
    if (!template) {
      setStatus("Select an AI model provider first.");
      return null;
    }
    const modelId = endpointDraft.modelId.trim() || wizard.model;
    if (!wizard.name.trim()) {
      setStatus("Enter an AI model name.");
      return null;
    }
    if (!modelId) {
      setStatus("Enter the provider-native model ID.");
      return null;
    }
    if (!["healthy", "rate_limited"].includes(endpointTestResult?.status)) {
      setStatus("Test the connection and model before saving the AI model.");
      return null;
    }
    setIsSaving(true);
    setStatus(`Adding ${wizard.name}...`);
    try {
      const connection = await saveEndpointConnection();
      const deployment = await createModelDeploymentFromTemplate(modelFarmWizardPayload(selectedTemplateId, {
        ...wizard,
        connectionId: connection.id,
        model: modelId,
        apiBase: "",
        credentialEnvRefs: {},
        credentialSecrets: {},
        metadata: {
          ...(wizard.metadata || {}),
          description: wizard.description || "",
          endpoint_name: connection.name,
          endpoint_url_configured: Boolean(connection.api_base)
        }
      }));
      setCreatedDeploymentId(deployment.id);
      if (endpointTestResult?.status === "rate_limited") {
        await refreshModelFarm();
        setStatus("AI model saved but not enabled because the upstream provider is temporarily rate-limited. Retry the model test later.");
        return deployment;
      }
      const tested = await testModelDeployment(deployment.id);
      await refreshModelFarm();
      setStatus(tested.status === "healthy"
        ? "AI model saved and verified. It can now be enabled."
        : `AI model saved, but its verification failed: ${tested.error || "unknown error"}`);
      return tested.deployment || deployment;
    } catch (error) {
      setStatus(error.message);
      return null;
    } finally {
      setIsSaving(false);
    }
  }

  async function testEndpointDraft() {
    const modelId = endpointDraft.modelId.trim() || wizard.model;
    if (!wizard.name.trim()) {
      setEndpointTestResult({ status: "unavailable", error: "Enter an AI model name before testing." });
      return;
    }
    if (!modelId) {
      setEndpointTestResult({ status: "unavailable", error: "Enter the provider-native model ID before testing." });
      return;
    }
    setIsTesting(true);
    setEndpointTestResult({ status: "testing", message: `Testing ${wizard.name}...` });
    try {
      const connection = await saveEndpointConnection();
      const connectionResult = await testModelConnection(connection.id);
      if (connectionResult.status !== "healthy") {
        throw new Error(connectionResult.error || "Connection test failed.");
      }
      const discovered = await listConnectionModels(connection.id).catch(() => connectionResult.sample_models || []);
      setAvailableModels(discovered || []);
      const draft = {
        ...wizard,
        connectionId: connection.id,
        model: modelId,
        apiBase: "",
        credentialEnvRefs: {},
        credentialSecrets: {},
        metadata: {
          ...(wizard.metadata || {}),
          description: wizard.description || "",
          endpoint_name: connection.name,
          endpoint_url_configured: Boolean(connection.api_base)
        }
      };
      const deploymentId = editingDeployment?.id || createdDeploymentId;
      const payload = deploymentId
        ? { deployment_id: deploymentId, template_id: selectedTemplateId, ...modelFarmDeploymentPatchPayload(draft) }
        : modelFarmWizardPayload(selectedTemplateId, draft);
      const result = await testModelDeploymentDraft(payload);
      const resultStatus = result.status === "healthy"
        ? "healthy"
        : result.status === "rate_limited"
          ? "rate_limited"
          : "unavailable";
      setEndpointTestResult({
        status: resultStatus,
        message: resultStatus === "healthy"
          ? "Connection and model test passed."
          : resultStatus === "rate_limited"
            ? "Connection passed; the selected model is temporarily rate-limited."
            : "Model test failed.",
        error: result.error || result.deployment?.last_error || "",
        retryable: Boolean(result.retryable),
        sample: result.sample || "",
        runtime: result.runtime || ""
      });
    } catch (error) {
      setEndpointTestResult({ status: "unavailable", message: "Endpoint test failed.", error: error.message });
    } finally {
      setIsTesting(false);
    }
  }

  async function addEndpointAndClose() {
    if (editingDeployment) {
      await updateExistingDeployment({ closeAfter: true });
      return;
    }
    const deployment = await createEndpointDeployment();
    if (deployment) {
      setIsAddOpen(false);
      setStatus("AI model added. Enable it from the card list when ready.");
    }
  }

  async function saveProviderDetails() {
    if (editingDeployment) {
      await updateExistingDeployment({ closeAfter: true });
      return;
    }
    if (createdDeploymentId) {
      closeAddModel();
      setStatus("AI model saved. Enable it from the card list when ready.");
      return;
    }
    if (!endpointDraft.modelId.trim()) {
      setStatus("Add at least one endpoint before saving this AI model.");
      setAddStage("endpoint");
      return;
    }
    await addEndpointAndClose();
  }

  async function updateExistingDeployment({ testAfter = false, closeAfter = false } = {}) {
    const deploymentId = editingDeployment?.id || createdDeploymentId;
    if (!deploymentId) {
      setStatus("Select an AI model before saving changes.");
      return null;
    }
    if (!wizard.name.trim()) {
      setStatus("Enter an AI model name.");
      return null;
    }
    setIsSaving(true);
    setStatus(testAfter ? `Saving and testing ${wizard.name}...` : `Saving ${wizard.name}...`);
    try {
      const connection = await saveEndpointConnection();
      const payload = modelFarmDeploymentPatchPayload({
        ...wizard,
        connectionId: connection.id,
        apiBase: "",
        credentialEnvRefs: {},
        credentialSecrets: {},
        metadata: {
          ...(wizard.metadata || {}),
          description: wizard.description || "",
          endpoint_name: connection.name,
          endpoint_url_configured: Boolean(connection.api_base)
        }
      });
      const updated = await updateModelDeployment(deploymentId, payload);
      setEditingDeployment(updated);
      setCreatedDeploymentId(updated.id);
      setWizard(modelFarmWizardFromDeployment(updated));
      setEndpointDraft((current) => ({ ...endpointDraftFromDeployment(updated), apiKey: current.apiKey }));
      setEndpointTestResult(null);
      await refreshModelFarm();
      if (testAfter) {
        setIsTesting(true);
        const result = await testModelDeployment(updated.id);
        setStatus(`${updated.name} endpoint test ${result.status || "completed"}.`);
        await refreshModelFarm();
      } else {
        setStatus(`${updated.name} updated.`);
      }
      if (closeAfter) closeAddModel();
      return updated;
    } catch (error) {
      setStatus(error.message);
      return null;
    } finally {
      setIsSaving(false);
      setIsTesting(false);
    }
  }


  async function runDeploymentTest(deployment) {
    setStatus(`Testing ${deployment.name}...`);
    try {
      const result = await testModelDeployment(deployment.id);
      if (result.status === "rate_limited") {
        setStatus(`${deployment.name} reached a temporary upstream rate limit. The connection remains valid; retry later or select another provider model.`);
      } else if (result.status === "healthy") {
        setStatus(`${deployment.name} test passed.`);
      } else {
        setStatus(`${deployment.name} test failed: ${result.error || "unknown provider error"}`);
      }
      await refreshModelFarm();
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function toggleDeployment(deployment) {
    setStatus(`${deployment.enabled ? "Disabling" : "Enabling"} ${deployment.name}...`);
    try {
      await updateModelDeployment(deployment.id, { enabled: !deployment.enabled });
      setStatus("AI model updated.");
      await refreshModelFarm();
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function removeDeployment(deployment) {
    const confirmed = await confirmAction({
      title: "Delete AI model?",
      message: `Delete "${deployment.name}"?`,
      detail: "This removes the deployment and its Model Farm usage/error history. Built-in local models cannot be deleted.",
      confirmLabel: "Delete AI model"
    });
    if (!confirmed) return;
    setStatus(`Deleting ${deployment.name}...`);
    try {
      await deleteModelDeployment(deployment.id);
      setStatus("AI model deleted.");
      await refreshModelFarm();
    } catch (error) {
      setStatus(error.message);
    }
  }

  const selectedTemplate = templates.find((template) => template.id === selectedTemplateId) || (editingDeployment ? templateFromDeployment(editingDeployment) : null);
  const isEditingDeployment = Boolean(editingDeployment);
  const providerOptions = providerTemplatesForTab(templates, providerTab, providerSearch);
  const selectedConnectionProvider = connectionProviderForTemplate(selectedTemplate || {});
  const compatibleConnections = connections.filter((connection) => connection.provider === selectedConnectionProvider);
  const visibleDeployments = deployments.filter((deployment) => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return true;
    return [
      deployment.name,
      deployment.provider,
      deployment.model,
      deployment.locality,
      ...(deployment.capabilities || [])
    ].join(" ").toLowerCase().includes(query);
  });
  const deploymentGroups = [
    { id: "experimentation", title: "Experimentation", description: "OpenRouter deployments for model comparison and rapid experiments." },
    { id: "production", title: "Production direct", description: "Direct OpenAI and Gemini provider credentials and billing." },
    { id: "local", title: "Local", description: "In-process models and self-hosted Ollama or vLLM endpoints." }
  ].map((group) => ({
    ...group,
    items: visibleDeployments.filter((deployment) => (
      (deployment.access_path || (deployment.locality === "local" ? "local" : "production")) === group.id
    ))
  }));

  return (
    <section className="page-stack ai-models-page">
      <header className="ai-models-header">
        <div>
          <h1>AI models</h1>
          <p>
            Power up apps with large language models and configure multiple endpoints under any supported provider.
            {" "}
            <a href="https://docs.litellm.ai/" target="_blank" rel="noreferrer">Learn more <IconOnly icon={ExternalLink} size={16} /></a>
          </p>
        </div>
        <button className="primary-action add-ai-model-button" type="button" onClick={openAddModel}>
          <IconLabel icon={Plus}>Add AI model</IconLabel>
        </button>
      </header>

      <div className="model-farm-summary ai-models-summary">
        <Metric label="Deployments" value={String(visibleDeployments.length)} />
        <Metric label="Monthly cost" value={`$${Number(usageSummary?.estimated_cost_usd || 0).toFixed(4)}`} />
        <Metric label="Attempts" value={String(usageSummary?.calls || 0)} />
        <Metric label="Failures" value={String(usageSummary?.failed_calls || 0)} />
      </div>

      <div className="ai-models-toolbar">
        <label className="ai-model-search">
          <IconOnly icon={Search} size={18} />
          <input value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} placeholder="Search" aria-label="Search AI models" />
        </label>
        <SelectField
          value={capabilityFilter}
          options={[{ value: "", label: "All capabilities" }, ...modelCapabilities.map((capability) => ({ value: capability, label: capability }))]}
          onChange={setCapabilityFilter}
        />
        <button className="secondary-action" type="button" onClick={refreshModelFarm}><IconLabel icon={RefreshCw}>Refresh</IconLabel></button>
      </div>

      {status && <div className="inline-status">{status}</div>}

      <section className="ai-model-groups">
        {isLoading && <p className="muted-text">Loading AI models...</p>}
        {!isLoading && visibleDeployments.length === 0 && <p className="muted-text">No AI models match this search.</p>}
        {deploymentGroups.filter((group) => group.items.length > 0).map((group) => (
          <section className="ai-model-group" key={group.id}>
            <header className="ai-model-group-header">
              <div><h2>{group.title}</h2><p>{group.description}</p></div>
              <span>{group.items.length}</span>
            </header>
            <div className="ai-model-card-grid">
              {group.items.map((deployment) => (
                <article className="ai-model-card" key={deployment.id}>
                  <header>
                    <ProviderLogo item={deployment} className={`ai-model-icon ${deployment.locality === "local" ? "local" : ""}`} />
                    <div>
                      <h2>{deployment.name}</h2>
                      <p>{providerLabelFromDeployment(deployment)}</p>
                    </div>
                    <button
                      className="ai-model-menu"
                      type="button"
                      disabled={Boolean(deployment.metadata?.builtin)}
                      aria-label={`Edit ${deployment.name}`}
                      onClick={() => openEditModel(deployment)}
                    >...</button>
                  </header>
                  <p className="ai-model-description">
                    {deployment.metadata?.description
                      || (deployment.metadata?.builtin
                        ? "Built-in local model available for development and offline demos."
                        : `${deployment.connection_name || "Provider connection"} · ${deployment.health_status}`)}
                  </p>
                  <p className="ai-model-version">Model: {truncateMiddle(deployment.model, 44)}</p>
                  <div className="deployment-chip-row">
                    {deployment.capabilities.slice(0, 4).map((capability) => <span key={capability}>{capability}</span>)}
                  </div>
                  <footer>
                    <span className={`ai-model-badge ${deployment.enabled ? "enabled" : "disabled"}`}>{deployment.enabled ? "Enabled" : "Disabled"}</span>
                    <div className="deployment-actions">
                      <button className="secondary-action" type="button" onClick={() => runDeploymentTest(deployment)}><IconLabel icon={Activity}>Test</IconLabel></button>
                      <button
                        className="secondary-action"
                        type="button"
                        disabled={!deployment.enabled && deployment.locality !== "local" && deployment.health_status !== "healthy"}
                        onClick={() => toggleDeployment(deployment)}
                      >
                        <IconLabel icon={CheckCircle2}>{deployment.enabled ? "Disable" : "Enable"}</IconLabel>
                      </button>
                      <button className="secondary-action danger-action" type="button" onClick={() => removeDeployment(deployment)} disabled={deployment.metadata?.builtin}>
                        <IconLabel icon={Trash2}>Delete</IconLabel>
                      </button>
                    </div>
                  </footer>
                </article>
              ))}
            </div>
          </section>
        ))}
      </section>

      <div className="ai-models-footer">
        <span>{visibleDeployments.length ? `1 to ${visibleDeployments.length} of ${visibleDeployments.length} items` : "0 items"}</span>
        <span className="ai-models-page-number">1</span>
      </div>

      {isAddOpen && (
        <div className="ai-model-modal-backdrop">
          {addStage === "select" && (
            <section className="ai-provider-picker" role="dialog" aria-modal="true" aria-label="Select an AI model provider">
              <header>
                <div>
                  <h2>Select a provider</h2>
                  <p className="ai-model-stepper">1. Select provider / 2. Provider details / 3. Add endpoint</p>
                </div>
                <button className="panel-collapse-button" type="button" onClick={closeAddModel} aria-label="Close provider picker">
                  <IconOnly icon={X} />
                </button>
              </header>
              <div className="ai-provider-picker-toolbar">
                <label className="ai-model-search">
                  <IconOnly icon={Search} size={18} />
                  <input value={providerSearch} onChange={(event) => setProviderSearch(event.target.value)} placeholder="Search" aria-label="Search providers" />
                </label>
                <div className="ai-provider-tabs" role="tablist" aria-label="AI model source">
                  <button type="button" className={providerTab === "provider" ? "active" : ""} onClick={() => setProviderTab("provider")}>LLM Provider</button>
                  <button type="button" className={providerTab === "local" ? "active" : ""} onClick={() => setProviderTab("local")}>Local LLM</button>
                </div>
              </div>
              <div className="ai-provider-picker-body">
                <p>{providerTab === "provider" ? "Add your own model from supported providers." : "Use built-in local models already registered for offline demos."}</p>
                <div className="ai-provider-grid">
                  {providerOptions.map((template) => (
                    <button
                      className={`ai-provider-option ${template.id === selectedTemplateId ? "selected" : ""}`}
                      key={template.id}
                      type="button"
                      onClick={() => applyTemplate(template)}
                    >
                      <ProviderLogo item={template} />
                      <strong>{template.creatable ? (template.provider_label || template.label) : template.label}</strong>
                    </button>
                  ))}
                  {!providerOptions.length && <p className="muted-text">No providers match this search.</p>}
                </div>
              </div>
              <footer>
                <button className="secondary-action" type="button" onClick={closeAddModel}>Cancel</button>
                <button className="primary-action" type="button" disabled={!selectedTemplateId} onClick={confirmProvider}>Confirm</button>
              </footer>
            </section>
          )}

          {addStage === "details" && selectedTemplate && (
            <section className="ai-model-detail-panel" role="dialog" aria-modal="true" aria-label={`Configure ${selectedTemplate.provider_label || selectedTemplate.label}`}>
              <header className="ai-model-detail-header">
                <div>
                  <button className="text-action" type="button" onClick={isEditingDeployment ? closeAddModel : () => setAddStage("select")}><IconLabel icon={ChevronLeft}>AI models</IconLabel></button>
                  <h2><ProviderLogo item={selectedTemplate} />{selectedTemplate.provider_label || selectedTemplate.label}</h2>
                  <a href="https://docs.litellm.ai/docs/providers" target="_blank" rel="noreferrer">Learn more about AI models <IconOnly icon={ExternalLink} size={16} /></a>
                  <p className="ai-model-stepper">{isEditingDeployment ? "Edit AI model details and endpoint settings." : "1. Select provider / 2. Provider details / 3. Add endpoint"}</p>
                </div>
                <div className="ai-model-detail-actions">
                  <button className="secondary-action" type="button" onClick={closeAddModel}>Cancel</button>
                  <button className="primary-action" type="button" onClick={saveProviderDetails} disabled={isSaving}>
                    <IconLabel icon={Save}>{isSaving ? "Saving..." : "Save"}</IconLabel>
                  </button>
                </div>
              </header>

              <section className="ai-model-detail-card">
                <h3>Details</h3>
                <label>
                  Name <span className="required-marker">*</span>
                  <input value={wizard.name} onChange={(event) => setWizard({ ...wizard, name: event.target.value })} />
                </label>
                <label>
                  Description
                  <textarea value={wizard.description || ""} onChange={(event) => setWizard({ ...wizard, description: event.target.value })} />
                </label>
              </section>

              <section className="ai-model-detail-card">
                <h3>Configuration</h3>
                <div className="ai-model-dev-tab">Development</div>
                <div className="ai-model-endpoints">
                  <h4>Endpoints</h4>
                  <p>{isEditingDeployment ? "Modify endpoint URL or replace the stored API key. Model ID is fixed after registration." : "Add at least 1 endpoint for the selected model or multiples to manage model load balancing."}</p>
                  {createdDeploymentId || endpointDraft.modelId ? (
                    <div className="ai-model-endpoint-row">
                      <div>
                        <strong>{endpointDraft.name || wizard.name || "Primary endpoint"}</strong>
                        <span>{endpointDraft.modelId || wizard.model}</span>
                      </div>
                      <button className="secondary-action" type="button" onClick={() => setAddStage("endpoint")}>
                        <IconLabel icon={Pencil}>Edit</IconLabel>
                      </button>
                    </div>
                  ) : (
                    <button className="text-action endpoint-add-button" type="button" onClick={() => setAddStage("endpoint")}>
                      <IconLabel icon={Plus}>Add</IconLabel>
                    </button>
                  )}
                </div>
                <div className="ai-model-usage-limit">
                  <h4>Usage limit</h4>
                  <p>Optionally, set a monthly budget to control the model consumption.</p>
                  <label>
                    <input type="number" min="0" step="0.01" value={wizard.monthlyBudgetUsd} onChange={(event) => setWizard({ ...wizard, monthlyBudgetUsd: Number(event.target.value) })} placeholder="Enter limit" />
                    <span>/ month</span>
                  </label>
                </div>
              </section>
            </section>
          )}

          {addStage === "endpoint" && selectedTemplate && (
            <section className="ai-endpoint-modal" role="dialog" aria-modal="true" aria-label="Add endpoint">
              <header>
                <div>
                  <button className="text-action" type="button" onClick={() => setAddStage("details")}><IconLabel icon={ChevronLeft}>{selectedTemplate.provider_label || selectedTemplate.label}</IconLabel></button>
                  <h2>{isEditingDeployment ? "Edit endpoint" : "Add endpoint"}</h2>
                  <a href="https://docs.litellm.ai/docs/providers" target="_blank" rel="noreferrer">Learn about {selectedTemplate.provider_label || selectedTemplate.label} connection <IconOnly icon={ExternalLink} size={16} /></a>
                  <p className="ai-model-stepper">{isEditingDeployment ? "Save endpoint changes or test the updated connection." : "1. Select provider / 2. Provider details / 3. Add endpoint"}</p>
                </div>
                <button className="panel-collapse-button" type="button" onClick={closeAddModel} aria-label="Close endpoint setup"><IconOnly icon={X} /></button>
              </header>
              <div className="ai-endpoint-form">
                <label>
                  Provider connection
                  <select
                    value={endpointDraft.connectionId || ""}
                    disabled={isEditingDeployment}
                    onChange={(event) => {
                      const connection = connections.find((item) => item.id === event.target.value);
                      updateEndpointDraft(connection
                        ? { connectionId: connection.id, name: connection.name, url: connection.api_base, apiKey: "" }
                        : { connectionId: "", name: "", url: selectedTemplate.deployment_defaults?.api_base || "", apiKey: "" });
                      setAvailableModels([]);
                    }}
                  >
                    <option value="">Create a new {selectedTemplate.provider_label || selectedTemplate.label} connection</option>
                    {compatibleConnections.map((connection) => (
                      <option key={connection.id} value={connection.id}>
                        {connection.name} · {connection.health_status}
                      </option>
                    ))}
                  </select>
                  <span>Connections keep endpoint credentials separate from reusable model deployments.</span>
                </label>
                <label>
                  Connection name <span className="required-marker">*</span>
                  <input value={endpointDraft.name} onChange={(event) => updateEndpointDraft({ name: event.target.value })} />
                </label>
                <label>
                  Model ID <span className="required-marker">*</span>
                  <input
                    list="available-model-ids"
                    value={endpointDraft.modelId}
                    readOnly={isEditingDeployment}
                    onChange={(event) => updateEndpointDraft({ modelId: event.target.value })}
                  />
                  <datalist id="available-model-ids">
                    {availableModels.map((model) => <option key={model.id} value={model.id}>{model.name || model.id}</option>)}
                  </datalist>
                  <span>{isEditingDeployment ? "Model ID is immutable. Delete and recreate this AI model to change it." : "Use a provider-native model ID. Successful connection tests load available model suggestions."}</span>
                </label>
                <label>
                  URL {requiresEndpointUrl(selectedTemplate) && <span className="required-marker">*</span>}
                  <input value={endpointDraft.url} onChange={(event) => updateEndpointDraft({ url: event.target.value })} placeholder={selectedTemplate.deployment_defaults?.api_base || "Optional provider base URL"} />
                </label>
                <label>
                  API key {selectedTemplate.credential_fields?.includes("api_key") && <span className="required-marker">*</span>}
                  <div className="secret-input">
                    <input
                      type={showEndpointSecret ? "text" : "password"}
                      value={endpointDraft.apiKey}
                      onChange={(event) => updateEndpointDraft({ apiKey: event.target.value })}
                      placeholder={isEditingDeployment ? "Leave blank to keep existing stored key" : ""}
                      autoComplete="off"
                    />
                    <button type="button" onClick={() => setShowEndpointSecret(!showEndpointSecret)} aria-label={showEndpointSecret ? "Hide API key" : "Show API key"}>
                      <IconOnly icon={showEndpointSecret ? EyeOff : Eye} size={18} />
                    </button>
                  </div>
                  <span>The API key is encrypted by the backend and is never returned to this browser.</span>
                </label>
                {endpointTestResult && (
                  <div className={`endpoint-test-result ${endpointTestResult.status === "healthy" ? "is-pass" : endpointTestResult.status === "rate_limited" ? "is-warning" : endpointTestResult.status === "testing" ? "is-testing" : "is-fail"}`}>
                    <IconOnly icon={endpointTestResult.status === "healthy" ? CheckCircle2 : endpointTestResult.status === "testing" ? RefreshCw : AlertTriangle} size={18} />
                    <div>
                      <strong>{endpointTestResult.message || (endpointTestResult.status === "healthy" ? "Endpoint test passed." : "Endpoint test failed.")}</strong>
                      {endpointTestResult.runtime && <span>Runtime: {endpointTestResult.runtime}</span>}
                      {endpointTestResult.sample && <span>Sample: {endpointTestResult.sample}</span>}
                      {endpointTestResult.error && <span>{endpointTestResult.error}</span>}
                    </div>
                  </div>
                )}
              </div>
              <footer>
                <button className="secondary-action" type="button" onClick={() => setAddStage("details")}>Back</button>
                <button className="secondary-action" type="button" onClick={testEndpointDraft} disabled={isSaving || isTesting}>
                  <IconLabel icon={Activity}>{isTesting ? "Testing..." : "Test endpoint"}</IconLabel>
                </button>
                <button
                  className="primary-action"
                  type="button"
                  onClick={addEndpointAndClose}
                  disabled={isSaving || isTesting || !["healthy", "rate_limited"].includes(endpointTestResult?.status)}
                >
                  <IconLabel icon={Save}>{isSaving ? "Saving..." : "Save endpoint"}</IconLabel>
                </button>
              </footer>
            </section>
          )}
        </div>
      )}
    </section>
  );
}

function EvaluationScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase, onOpenDetail, confirmAction }) {
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [chatConfigurations, setChatConfigurations] = useState([]);
  const [judgeDeployments, setJudgeDeployments] = useState([]);
  const [runs, setRuns] = useState([]);
  const [cases, setCases] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [form, setForm] = useState({
    knowledgeBaseId: selectedKnowledgeBaseId || "",
    chatConfigurationId: "",
    judgeDeploymentId: "",
    retrievalMode: "hybrid",
    topK: 4,
    limit: 20,
    compareBaseline: true,
    runRagxplain: false
  });
  const [status, setStatus] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadEvaluationData() {
      setIsLoading(true);
      try {
        const [nextKnowledgeBases, nextConfigurations, nextRuns, nextJudges] = await Promise.all([
          listKnowledgeBases(),
          listChatConfigurations(),
          listEvaluationRuns(),
          listModelDeployments({ capability: "judge", enabled: true }).catch(() => [])
        ]);
        if (cancelled) return;
        setKnowledgeBases(nextKnowledgeBases);
        setChatConfigurations(nextConfigurations);
        setRuns(nextRuns);
        setJudgeDeployments(nextJudges);
        setForm((current) => ({
          ...current,
          knowledgeBaseId: current.knowledgeBaseId || selectedKnowledgeBaseId || nextKnowledgeBases[0]?.id || "",
          chatConfigurationId: current.chatConfigurationId || nextConfigurations[0]?.id || "",
          judgeDeploymentId: current.judgeDeploymentId || nextJudges[0]?.id || ""
        }));
        setSelectedRunId((current) => current || nextRuns[0]?.id || "");
      } catch (error) {
        if (!cancelled) setStatus(`Evaluation data load failed: ${error.message}`);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }
    loadEvaluationData();
    return () => { cancelled = true; };
  }, [selectedKnowledgeBaseId]);

  useEffect(() => {
    let cancelled = false;
    async function loadCases() {
      if (!selectedRunId) {
        setCases([]);
        return;
      }
      try {
        const nextCases = await listEvaluationCases(selectedRunId);
        if (!cancelled) setCases(nextCases);
      } catch (error) {
        if (!cancelled) setStatus(`Evaluation case load failed: ${error.message}`);
      }
    }
    loadCases();
    return () => { cancelled = true; };
  }, [selectedRunId]);

  const selectedKnowledgeBase = knowledgeBases.find((item) => item.id === form.knowledgeBaseId);
  const selectedRun = runs.find((run) => run.id === selectedRunId);

  async function refreshRuns(nextSelectedRunId = selectedRunId) {
    const nextRuns = await listEvaluationRuns();
    setRuns(nextRuns);
    setSelectedRunId(nextSelectedRunId || nextRuns[0]?.id || "");
  }

  async function startEvaluation() {
    if (!form.knowledgeBaseId) {
      setStatus("Select a knowledge base before running evaluation.");
      return;
    }
    setIsRunning(true);
    setStatus("Running evaluation...");
    try {
      const run = await createEvaluationRun({
        name: `Adaptive vs Static L2 - ${new Date().toLocaleString()}`,
        knowledge_base_id: form.knowledgeBaseId,
        chat_configuration_id: form.chatConfigurationId || null,
        judge_deployment_id: form.judgeDeploymentId || "",
        retrieval_mode: form.retrievalMode,
        top_k: Number(form.topK),
        limit: Number(form.limit),
        compare_baseline: form.compareBaseline,
        run_ragxplain: form.runRagxplain
      });
      onSelectKnowledgeBase(form.knowledgeBaseId);
      await refreshRuns(run.id);
      const ragxplainStatus = run.metadata?.ragxplain?.status;
      if (ragxplainStatus === "failed") {
        setStatus(`Evaluation completed, but RAGXplain failed: ${run.metadata.ragxplain.error}`);
      } else {
        const suffix = ragxplainStatus === "completed" ? " RAGXplain insights are ready." : "";
        setStatus(`Evaluation completed: ${run.metadata?.record_count || run.limit} case(s).${suffix}`);
      }
    } catch (error) {
      setStatus(`Evaluation failed: ${error.message}`);
    } finally {
      setIsRunning(false);
    }
  }

  async function removeSelectedRun() {
    if (!selectedRun) return;
    const confirmed = await confirmAction({
      title: "Delete evaluation run?",
      message: `Delete evaluation run "${selectedRun.name}"?`,
      detail: "Stored case results, trace metadata, and RAGXplain artifacts for this run will be removed.",
      confirmLabel: "Delete run"
    });
    if (!confirmed) return;
    try {
      await deleteEvaluationRun(selectedRun.id);
      setCases([]);
      await refreshRuns("");
      setStatus("Evaluation run deleted.");
    } catch (error) {
      setStatus(`Delete evaluation run failed: ${error.message}`);
    }
  }

  return (
    <section className="evaluation-grid">
      <section className="panel evaluation-control-panel">
        <PanelHeader eyebrow="Dataset" title="Benchmark setup" />
        <Metric label="Dataset" value="WixQA / processed QAC" />
        <Metric label="Default limit" value="20 cases" />
        <Metric label="Max sync run" value="100 cases" />
        <Metric label="Labels" value="simple / moderate / complex" />
        <div className="config-section runtime-section evaluation-form">
          <SelectField
            label="Knowledge base"
            value={form.knowledgeBaseId}
            options={[{ value: "", label: "Select Knowledge Base" }, ...knowledgeBases.map((item) => ({ value: item.id, label: item.name }))]}
            onChange={(knowledgeBaseId) => {
              setForm({ ...form, knowledgeBaseId });
              onSelectKnowledgeBase(knowledgeBaseId);
            }}
          />
          <SelectField
            label="Chat configuration"
            value={form.chatConfigurationId}
            options={[{ value: "", label: "Default configuration" }, ...chatConfigurations.map((item) => ({ value: item.id, label: item.name }))]}
            onChange={(chatConfigurationId) => setForm({ ...form, chatConfigurationId })}
          />
          <SelectField
            label="Judge deployment"
            value={form.judgeDeploymentId}
            options={[
              { value: "", label: "No registered judge selected" },
              ...judgeDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(judgeDeploymentId) => setForm({ ...form, judgeDeploymentId })}
          />
          <div className="config-two-column">
            <SelectField
              label="Retrieval mode"
              value={form.retrievalMode}
              options={[{ value: "hybrid", label: "Hybrid" }, { value: "bm25", label: "BM25" }, { value: "dense", label: "Dense" }]}
              onChange={(retrievalMode) => setForm({ ...form, retrievalMode })}
            />
            <label>
              Top K
              <input type="number" min="1" max="50" value={form.topK} onChange={(event) => setForm({ ...form, topK: Number(event.target.value) })} />
            </label>
          </div>
          <div className="config-two-column">
            <label>
              Dataset limit
              <input type="number" min="0" max="100" value={form.limit} onChange={(event) => setForm({ ...form, limit: Number(event.target.value) })} />
            </label>
            <label className="check-row evaluation-check-row">
              <input
                type="checkbox"
                checked={form.compareBaseline}
                onChange={(event) => setForm({ ...form, compareBaseline: event.target.checked })}
              />
              Compare static L2
            </label>
          </div>
          <label className="check-row evaluation-check-row ragxplain-toggle">
            <input
              type="checkbox"
              checked={form.runRagxplain}
              onChange={(event) => setForm({ ...form, runRagxplain: event.target.checked })}
            />
            <span><strong>Run RAGXplain LLM Judge</strong><small>Generate executive insights and prioritized actions.</small></span>
          </label>
          {selectedKnowledgeBase && (
            <dl className="source-facts evaluation-kb-summary">
              <div><dt>Status</dt><dd>{selectedKnowledgeBase.status}</dd></div>
              <div><dt>Documents</dt><dd>{selectedKnowledgeBase.document_count}</dd></div>
              <div><dt>Chunks</dt><dd>{selectedKnowledgeBase.chunk_count}</dd></div>
              <div><dt>Embedding</dt><dd>{selectedKnowledgeBase.embedding_model || "-"}</dd></div>
            </dl>
          )}
          <button className="primary-action" type="button" onClick={startEvaluation} disabled={isRunning || isLoading}>
            <IconLabel icon={isRunning ? RefreshCw : ClipboardList}>{isRunning ? "Running..." : "Run evaluation"}</IconLabel>
          </button>
          {status && <p className="muted-text compact-muted">{status}</p>}
        </div>
      </section>
      <section className="panel evaluation-results-panel">
        <PanelHeader eyebrow="Evaluation" title="Runs & results" />
        {runs.length === 0 ? (
          <div className="empty-state"><strong>No evaluation runs yet</strong><p>Run a bounded benchmark to compare Adaptive RAG with static L2 Simple RAG.</p></div>
        ) : (
          <div className="run-list evaluation-run-list">
            {runs.map((run) => (
              <article key={run.id} className={`run-card ${run.id === selectedRunId ? "selected" : ""}`}>
                <button type="button" className="run-card-select" onClick={() => setSelectedRunId(run.id)}>
                  <div>
                    <strong>{run.name}</strong>
                    <small>{run.dataset_name} - {run.status} - {run.metadata?.record_count ?? run.limit} cases</small>
                    <span className={`evaluation-ragxplain-status is-${run.metadata?.ragxplain?.status || "not_requested"}`}>
                      RAGXplain: {(run.metadata?.ragxplain?.status || "not_requested").replace("_", " ")}
                    </span>
                  </div>
                  <dl>
                    <Metric label="Routing" value={formatPercentMetric(run.metrics?.routing_accuracy)} />
                    <Metric label="Context" value={formatPercentMetric(run.metrics?.context_relevance)} />
                    <Metric label="Faithfulness" value={formatPercentMetric(run.metrics?.faithfulness_proxy)} />
                    <Metric label="Latency" value={`${formatNumber(run.metrics?.average_latency_ms)} ms`} />
                  </dl>
                </button>
              </article>
            ))}
          </div>
        )}
        {selectedRun && (
          <div className="evaluation-run-detail">
            <div className="action-row">
              <button
                className="secondary-action"
                type="button"
                onClick={() => onOpenDetail(selectedRun, null, "ragxplain")}
                disabled={selectedRun.metadata?.ragxplain?.status !== "completed"}
                title={selectedRun.metadata?.ragxplain?.error || "Open run-level RAGXplain insights"}
              >
                <IconLabel icon={GitBranch}>Open RAGXplain insights</IconLabel>
              </button>
              <button className="secondary-action danger-action" type="button" onClick={removeSelectedRun}>
                <IconLabel icon={Trash2}>Delete run</IconLabel>
              </button>
            </div>
            <div className={`evaluation-ragxplain-summary is-${selectedRun.metadata?.ragxplain?.status || "not_requested"}`}>
              <div>
                <strong>RAGXplain {(selectedRun.metadata?.ragxplain?.status || "not_requested").replace("_", " ")}</strong>
                <span>{selectedRun.metadata?.ragxplain?.judge || "No judge was requested for this run."}</span>
              </div>
              {selectedRun.metadata?.ragxplain?.error && <p>{selectedRun.metadata.ragxplain.error}</p>}
            </div>
            <div className="metrics-grid evaluation-metrics-grid">
              <article className="metric-card"><small>Adaptive routes</small><strong>{formatDistribution(selectedRun.route_distribution)}</strong><span>L1/L2/L3 distribution</span></article>
              <article className="metric-card"><small>Static baseline</small><strong>{formatPercentMetric(selectedRun.baseline_metrics?.answer_overlap)}</strong><span>Answer overlap</span></article>
              <article className="metric-card"><small>Runtime proxy</small><strong>{formatNumber(selectedRun.metrics?.runtime_proxy_units)}</strong><span>chars / 1k</span></article>
              <article className="metric-card"><small>Retrieved contexts</small><strong>{formatNumber(selectedRun.metrics?.average_retrieved_contexts)}</strong><span>average per case</span></article>
            </div>
            <div className="table-panel evaluation-case-table">
              <table>
                <thead>
                  <tr>
                    <th>Question</th>
                    <th>Label</th>
                    <th>Adaptive route</th>
                    <th>Context</th>
                    <th>Overlap</th>
                    <th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((evaluationCase) => (
                    <tr key={evaluationCase.id}>
                      <td>{evaluationCase.question}</td>
                      <td>{evaluationCase.complexity_label}</td>
                      <td>{evaluationCase.adaptive_metadata?.route_label || evaluationCase.adaptive_metadata?.route_level || "-"}</td>
                      <td>{formatPercentMetric(evaluationCase.metrics?.adaptive?.context_relevance)}</td>
                      <td>{formatPercentMetric(evaluationCase.metrics?.adaptive?.answer_overlap)}</td>
                      <td><button className="secondary-action compact-action" type="button" onClick={() => onOpenDetail(selectedRun, evaluationCase, "case")}><IconLabel icon={GitBranch}>Trace</IconLabel></button></td>
                    </tr>
                  ))}
                  {cases.length === 0 && (
                    <tr><td colSpan="6">{selectedRunId ? "Loading cases..." : "Select a run."}</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </section>
  );
}

function EvaluationDetailScreen({ detail, onBack }) {
  const evaluationCase = detail?.evaluationCase;
  const run = detail?.run;
  if (detail?.view === "ragxplain") {
    return <RagxplainInsightsScreen run={run} onBack={onBack} />;
  }
  const traceSteps = Array.isArray(evaluationCase?.adaptive_metadata?.trace_steps) ? evaluationCase.adaptive_metadata.trace_steps : [];
  if (!evaluationCase) {
    return (
      <section className="page-stack">
        <PanelHeader eyebrow="Evaluation Detail" title="RAGXplain" />
        <div className="empty-state"><strong>No evaluation case selected</strong><p>Open a case from the Evaluation screen to inspect its route, sources and trace.</p></div>
        <button className="secondary-action" type="button" onClick={onBack}><IconLabel icon={ChevronLeft}>Back to Evaluation</IconLabel></button>
      </section>
    );
  }
  return (
    <section className="page-stack evaluation-detail-page">
      <div className="action-row">
        <button className="secondary-action" type="button" onClick={onBack}><IconLabel icon={ChevronLeft}>Back to Evaluation</IconLabel></button>
      </div>
      <PanelHeader eyebrow="Evaluation Detail" title="RAGXplain" />
      <section className="panel evaluation-case-summary">
        <p className="eyebrow">{run?.name || "Evaluation run"}</p>
        <h2>{evaluationCase.question}</h2>
        <div className="metrics-grid evaluation-metrics-grid">
          <article className="metric-card"><small>Expected label</small><strong>{evaluationCase.complexity_label}</strong><span>benchmark</span></article>
          <article className="metric-card"><small>Adaptive route</small><strong>{evaluationCase.adaptive_metadata?.route_label || "-"}</strong><span>{evaluationCase.adaptive_metadata?.complexity_label || "classifier"}</span></article>
          <article className="metric-card"><small>Context relevance</small><strong>{formatPercentMetric(evaluationCase.metrics?.adaptive?.context_relevance)}</strong><span>proxy</span></article>
          <article className="metric-card"><small>Answer overlap</small><strong>{formatPercentMetric(evaluationCase.metrics?.adaptive?.answer_overlap)}</strong><span>expected answer</span></article>
        </div>
      </section>
      <section className="evaluation-answer-grid">
        <article className="panel">
          <h3>Adaptive answer</h3>
          <p>{evaluationCase.adaptive_answer}</p>
        </article>
        <article className="panel">
          <h3>Static L2 answer</h3>
          <p>{evaluationCase.static_answer || "Baseline disabled for this run."}</p>
        </article>
      </section>
      <section className="panel">
        <PanelHeader eyebrow="Sources" title="Adaptive retrieved contexts" />
        <div className="run-list">
          {evaluationCase.adaptive_contexts.map((context) => (
            <article className="run-card" key={context.id}>
              <div>
                <strong>{context.metadata?.title || context.id}</strong>
                <small>{context.mode} - rank {context.rank} - chunk {context.metadata?.chunk_index ?? "-"}</small>
              </div>
              <p>{context.text}</p>
            </article>
          ))}
          {evaluationCase.adaptive_contexts.length === 0 && <p className="muted-text">No adaptive contexts returned.</p>}
        </div>
      </section>
      <section className="panel">
        <PanelHeader eyebrow="Trace" title="Adaptive pipeline trace" />
        <div className="trace-board">
          {traceSteps.map((step, index) => (
            <article key={`${step.step}-${index}`}>
              <span>{index + 1}</span>
              <strong>{step.step}</strong>
              <p>{step.detail}</p>
            </article>
          ))}
          {traceSteps.length === 0 && <p className="muted-text">No trace metadata returned.</p>}
        </div>
      </section>
    </section>
  );
}

function RagxplainInsightsScreen({ run, onBack }) {
  const [reloadKey, setReloadKey] = useState(0);
  const ragxplain = run?.metadata?.ragxplain || {};
  const viewerUrl = run?.id ? getRagxplainViewerUrl(run.id) : "";

  if (!run) {
    return (
      <section className="page-stack evaluation-detail-page">
        <button className="secondary-action" type="button" onClick={onBack}><IconLabel icon={ChevronLeft}>Back to Evaluation</IconLabel></button>
        <div className="empty-state"><strong>No evaluation run selected</strong><p>Select a completed RAGXplain run from Evaluation.</p></div>
      </section>
    );
  }

  return (
    <section className="page-stack evaluation-detail-page ragxplain-insights-page">
      <header className="ragxplain-insights-header">
        <div>
          <p className="eyebrow">Evaluation Insights</p>
          <h1>{run.name}</h1>
          <p>Executive summary and prioritized action items from RAGXplain.</p>
        </div>
        <div className="action-row">
          <button className="secondary-action" type="button" onClick={onBack}><IconLabel icon={ChevronLeft}>Back to Evaluation</IconLabel></button>
          <button className="secondary-action" type="button" onClick={() => setReloadKey((current) => current + 1)} disabled={ragxplain.status !== "completed"}>
            <IconLabel icon={RotateCw}>Reload</IconLabel>
          </button>
          <a className="secondary-action" href={viewerUrl} target="_blank" rel="noreferrer">
            <IconLabel icon={ExternalLink}>Open in new tab</IconLabel>
          </a>
        </div>
      </header>
      <div className={`evaluation-ragxplain-summary is-${ragxplain.status || "not_requested"}`}>
        <div><strong>RAGXplain {String(ragxplain.status || "not_requested").replace("_", " ")}</strong><span>{ragxplain.judge || "Judge unavailable"}</span></div>
        {ragxplain.error && <p>{ragxplain.error}</p>}
      </div>
      {ragxplain.status === "completed" ? (
        <div className="ragxplain-viewer-frame">
          <iframe
            key={reloadKey}
            src={viewerUrl}
            title={`RAGXplain insights for ${run.name}`}
            sandbox="allow-scripts allow-same-origin allow-downloads"
          />
        </div>
      ) : (
        <div className="empty-state"><strong>Insights are not ready</strong><p>{ragxplain.error || "Run the RAGXplain judge for this evaluation first."}</p></div>
      )}
    </section>
  );
}

function AnalyticsScreen() {
  const [tab, setTab] = useState("tokens");
  const [usage, setUsage] = useState([]);
  const [summary, setSummary] = useState({});
  const [status, setStatus] = useState("");

  useEffect(() => {
    async function loadAnalytics() {
      try {
        const [nextUsage, nextSummary] = await Promise.all([
          listModelUsage({ limit: 200 }),
          getModelUsageSummary()
        ]);
        setUsage(nextUsage);
        setSummary(nextSummary);
        setStatus("");
      } catch (error) {
        setStatus(`Analytics unavailable: ${error.message}`);
      }
    }
    loadAnalytics();
  }, []);

  const realTokenStats = [
    { label: "Total attempts", value: summary.calls || usage.length, delta: "AI Models recorded calls" },
    { label: "Tokens", value: summary.total_tokens || usage.reduce((sum, event) => sum + Number(event.total_tokens || 0), 0), delta: "input + output" },
    { label: "Estimated cost", value: `$${formatNumber(summary.estimated_cost_usd || usage.reduce((sum, event) => sum + Number(event.estimated_cost_usd || 0), 0), 4)}`, delta: "current month" },
    { label: "Failures", value: summary.failed_calls || usage.filter((event) => event.status !== "completed").length, delta: "provider/runtime errors" }
  ];
  return (
    <section className="page-stack">
      <PanelHeader eyebrow="Usage Analytics" title="Analytics" />
      {status && <div className="inline-status">{status}</div>}
      <div className="tabs">
        <button className={tab === "tokens" ? "active" : ""} onClick={() => setTab("tokens")}><IconLabel icon={BarChart3}>Token statistics</IconLabel></button>
        <button className={tab === "feedback" ? "active" : ""} onClick={() => setTab("feedback")}><IconLabel icon={ThumbsUp}>Detailed Statistics & Feedbacks</IconLabel></button>
      </div>
      {tab === "tokens" ? (
        <>
          <div className="metrics-grid">
            {realTokenStats.map((stat) => (
              <article className="metric-card" key={stat.label}>
                <small>{stat.label}</small>
                <strong>{stat.value}</strong>
                <span>{stat.delta}</span>
              </article>
            ))}
          </div>
          <div className="table-panel">
            <table>
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Purpose</th>
                  <th>Provider</th>
                  <th>Model</th>
                  <th>Status</th>
                  <th>Tokens</th>
                  <th>Cost</th>
                </tr>
              </thead>
              <tbody>
                {usage.slice(0, 50).map((event) => (
                  <tr key={event.id}>
                    <td>{formatDateTime(event.created_at)}</td>
                    <td>{event.purpose}</td>
                    <td>{event.provider}</td>
                    <td>{event.model}</td>
                    <td>{event.status}</td>
                    <td>{event.total_tokens}</td>
                    <td>${formatNumber(event.estimated_cost_usd, 5)}</td>
                  </tr>
                ))}
                {usage.length === 0 && <tr><td colSpan="7">No model usage recorded yet.</td></tr>}
              </tbody>
            </table>
          </div>
        </>
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

function IconLabel({ icon: Icon, children, size = 16 }) {
  return (
    <span className="icon-label">
      <Icon className="button-icon" size={size} aria-hidden="true" strokeWidth={2} />
      <span>{children}</span>
    </span>
  );
}

function AssistantMessageContent({ content }) {
  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a: ({ children, href, title }) => (
            <a href={href} title={title} target="_blank" rel="noreferrer">
              {children}
            </a>
          )
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function IconOnly({ icon: Icon, size = 18 }) {
  return <Icon className="button-icon" size={size} aria-hidden="true" strokeWidth={2} />;
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
  const normalizedOptions = options.map((option) => (
    typeof option === "string" ? { value: option, label: option } : option
  ));
  return (
    <label>
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {normalizedOptions.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled} title={option.title || option.label}>{option.label}</option>
        ))}
      </select>
    </label>
  );
}

function chatConfigurationOption(configuration = {}) {
  const configurationCode = configurationCodeFromRecord(configuration);
  const name = configuration.name || "Untitled configuration";
  return {
    value: configuration.id,
    label: `${name} (${configurationCode})`,
    title: `${name} (${configurationCode})`
  };
}

function existingConfigurationCodes(configurations = []) {
  return new Set(configurations.map((configuration) => configurationCodeFromRecord(configuration)).filter(Boolean));
}

function configurationCodeFromRecord(configuration = {}) {
  return normalizeConfigurationCode(configuration.metadata?.configuration_id) || fallbackConfigurationCode(configuration.id);
}

function normalizeConfigurationCode(value) {
  const text = String(value || "").trim();
  if (text.length === CONFIGURATION_DISPLAY_ID_LENGTH && /^[A-Za-z0-9]+$/.test(text)) return text;
  return "";
}

function createConfigurationCode(existingCodes = new Set()) {
  const existing = existingCodes instanceof Set ? existingCodes : new Set(existingCodes);
  for (let attempt = 0; attempt < 20; attempt += 1) {
    let value = "";
    const bytes = new Uint8Array(CONFIGURATION_DISPLAY_ID_LENGTH);
    if (globalThis.crypto?.getRandomValues) {
      globalThis.crypto.getRandomValues(bytes);
      value = Array.from(bytes, (byte) => CONFIGURATION_DISPLAY_ID_ALPHABET[byte % CONFIGURATION_DISPLAY_ID_ALPHABET.length]).join("");
    } else {
      value = Array.from({ length: CONFIGURATION_DISPLAY_ID_LENGTH }, () => (
        CONFIGURATION_DISPLAY_ID_ALPHABET[Math.floor(Math.random() * CONFIGURATION_DISPLAY_ID_ALPHABET.length)]
      )).join("");
    }
    if (!existing.has(value)) return value;
  }
  return fallbackConfigurationCode(`${Date.now()}-${Math.random()}`);
}

function fallbackConfigurationCode(value = "") {
  let seed = 2166136261;
  const text = String(value || "configuration");
  for (let index = 0; index < text.length; index += 1) {
    seed ^= text.charCodeAt(index);
    seed = Math.imul(seed, 16777619) >>> 0;
  }
  let output = "";
  for (let index = 0; index < CONFIGURATION_DISPLAY_ID_LENGTH; index += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    output += CONFIGURATION_DISPLAY_ID_ALPHABET[seed % CONFIGURATION_DISPLAY_ID_ALPHABET.length];
  }
  return output;
}

function documentMatchesFilterQuery(document, query) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  if (!normalizedQuery) return true;
  const searchable = [
    document.title,
    document.text,
    document.metadata?.filename,
    document.metadata?.source_type,
    document.metadata?.mime_type,
    document.metadata?.uri,
    document.id
  ].filter(Boolean).join(" ").toLowerCase();
  return searchable.includes(normalizedQuery);
}

function compactPreview(value, maxLength = 120) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) return text || "No preview available.";
  return `${text.slice(0, Math.max(0, maxLength - 1)).trim()}...`;
}

function formatInteger(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0";
  return Math.max(0, Math.trunc(number)).toLocaleString();
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatPercentMetric(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${Math.round(number * 100)}%`;
}

function formatNumber(value, digits = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toFixed(digits).replace(/\.0$/, "");
}

function formatDistribution(distribution = {}) {
  const entries = Object.entries(distribution || {});
  if (entries.length === 0) return "-";
  return entries.map(([key, value]) => `${key.replace("l1_", "L1 ").replace("l2_", "L2 ").replace("l3_", "L3 ")}: ${value}`).join(" / ");
}

function answerModeFromRoute(route) {
  if (route === "L1 Direct") return "direct";
  if (route === "L2 Simple RAG") return "simple_rag";
  if (route === "L3 Complex RAG") return "complex_rag";
  return "adaptive";
}

function retrievalModeValue(value) {
  if (value === "BM25") return "bm25";
  if (value === "Dense") return "dense";
  return "hybrid";
}

function retrievalModeLabel(value) {
  if (value === "bm25") return "BM25";
  if (value === "dense") return "Dense";
  return "Hybrid";
}

function routeLabelFromMode(mode) {
  if (mode === "direct") return "L1 Direct";
  if (mode === "simple_rag") return "L2 Simple RAG";
  if (mode === "complex_rag") return "L3 Complex RAG";
  return "Adaptive";
}

function routeValues() {
  return routes.map((route) => route.value);
}

function defaultRuntimeRoute() {
  return routes[0]?.value || "Adaptive";
}

function retrievalModeLabels() {
  return ["Hybrid", "BM25", "Dense"];
}

function messagesFromChatRecords(records = []) {
  let previousUserQuestion = "";
  return records.map((record) => {
    if (record.role === "user") {
      previousUserQuestion = record.content;
      return {
        id: record.id,
        role: "user",
        content: record.content,
        contexts: [],
        metadata: record.metadata || {},
        status: record.status || "completed"
      };
    }
    return {
      id: record.id,
      role: "assistant",
      question: record.metadata?.question || previousUserQuestion,
      content: record.content,
      contexts: record.contexts || [],
      metadata: record.metadata || {},
      status: record.status || "completed",
      streaming: false
    };
  });
}

function updateMessage(messages, messageId, patchOrUpdater) {
  return messages.map((message) => {
    if (message.id !== messageId) return message;
    const patch = typeof patchOrUpdater === "function" ? patchOrUpdater(message) : patchOrUpdater;
    return { ...message, ...patch };
  });
}

function welcomeMessagesFromConfig(config = {}) {
  const content = String(config.welcomeMessage || "").trim();
  if (!content) return [];
  return [
    {
      id: "welcome-message",
      role: "assistant",
      content,
      metadata: {
        welcome: true,
        complexity_label: "ready",
        retrieval_mode: "none",
        top_k: 0,
        latency_ms: 0
      },
      contexts: [],
      status: "completed"
    }
  ];
}

function isWelcomeMessage(message = {}) {
  return Boolean(message.metadata?.welcome);
}

function normalizeConversationStarters(starters) {
  if (Array.isArray(starters)) {
    return starters.map((starter) => String(starter || "").trim()).filter(Boolean);
  }
  return parseConversationStarters(starters);
}

function parseConversationStarters(value) {
  return String(value || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function applyChatConfigurationToDraft(current, record) {
  return applyChatConfigurationSnapshotToDraft(current, chatConfigurationSnapshotFromRecord(record), record.id);
}

function applyChatConfigurationSnapshotToDraft(current, snapshot = {}, configurationId = "") {
  const provider = String(snapshot.generator_provider || defaultChatConfigurationDraft.generatorProvider).trim();
  const model = String(snapshot.generator_model || defaultChatConfigurationDraft.generatorModel).trim();
  const metadata = snapshot.metadata || {};
  const route = routeValues().includes(metadata.route_strategy) ? metadata.route_strategy : current.route;
  const retrievalMode = retrievalModeLabels().includes(metadata.retrieval_mode_label)
    ? metadata.retrieval_mode_label
    : retrievalModeLabel(metadata.retrieval_mode || current.retrievalMode);
  return {
    ...current,
    chatConfigurationId: configurationId || snapshot.id || "",
    configurationCode: normalizeConfigurationCode(metadata.configuration_id) || fallbackConfigurationCode(configurationId || snapshot.id),
    configurationCreatedAt: snapshot.created_at || "",
    configurationUpdatedAt: snapshot.updated_at || "",
    configurationName: snapshot.name || defaultChatConfigurationDraft.configurationName,
    configurationDescription: snapshot.description || "",
    route,
    classifier: metadata.classifier || current.classifier,
    classifierDeploymentId: metadata.classifier_deployment_id || "",
    queryEmbeddingDeploymentId: metadata.query_embedding_deployment_id || "",
    retrievalMode,
    topK: clampNumber(metadata.top_k, current.topK || 6, 1, 50),
    reranker: typeof metadata.reranker_enabled === "boolean" ? metadata.reranker_enabled : current.reranker,
    generatorDeploymentId: snapshot.generator_deployment_id || metadata.generator_deployment_id || defaultChatConfigurationDraft.generatorDeploymentId,
    fallbackDeploymentIds: Array.isArray(snapshot.fallback_deployment_ids) ? snapshot.fallback_deployment_ids : [],
    rerankerDeploymentId: snapshot.reranker_deployment_id || "",
    plannerDeploymentId: snapshot.planner_deployment_id || "",
    generationParameters: snapshot.generation_parameters || defaultChatConfigurationDraft.generationParameters,
    citationsEnabled: typeof snapshot.citations_enabled === "boolean" ? snapshot.citations_enabled : defaultChatConfigurationDraft.citationsEnabled,
    citations: typeof snapshot.citations_enabled === "boolean" ? snapshot.citations_enabled : current.citations,
    generatorProvider: provider,
    generatorModel: model,
    responseStructure: responseStructures.includes(snapshot.response_structure) ? snapshot.response_structure : defaultChatConfigurationDraft.responseStructure,
    tone: chatbotTones.includes(snapshot.tone) ? snapshot.tone : defaultChatConfigurationDraft.tone,
    humorLevel: clampNumber(snapshot.humor_level, defaultChatConfigurationDraft.humorLevel, 0, 5),
    welcomeMessage: metadata.welcome_message || defaultChatConfigurationDraft.welcomeMessage,
    conversationStarters: Array.isArray(metadata.conversation_starters)
      ? metadata.conversation_starters
      : defaultChatConfigurationDraft.conversationStarters,
    systemPrompt: snapshot.system_prompt || "",
    predefinedPrompt: snapshot.predefined_prompt || ""
  };
}

function chatConfigurationSnapshotFromRecord(record = {}) {
  return {
    id: record.id || "",
    name: record.name,
    description: record.description,
    generator_provider: record.generator_provider,
    generator_model: record.generator_model,
    response_structure: record.response_structure,
    tone: record.tone,
    humor_level: record.humor_level,
    system_prompt: record.system_prompt,
    predefined_prompt: record.predefined_prompt,
    generator_deployment_id: record.generator_deployment_id,
    fallback_deployment_ids: record.fallback_deployment_ids || [],
    reranker_deployment_id: record.reranker_deployment_id || "",
    planner_deployment_id: record.planner_deployment_id || "",
    generation_parameters: record.generation_parameters || {},
    citations_enabled: record.citations_enabled,
    created_at: record.created_at || "",
    updated_at: record.updated_at || "",
    metadata: record.metadata || {}
  };
}

function chatConfigurationPayloadFromDraft(config) {
  return {
    name: config.configurationName || defaultChatConfigurationDraft.configurationName,
    description: config.configurationDescription || "",
    generator_provider: config.generatorProvider || defaultChatConfigurationDraft.generatorProvider,
    generator_model: config.generatorModel || defaultChatConfigurationDraft.generatorModel,
    response_structure: config.responseStructure || defaultChatConfigurationDraft.responseStructure,
    tone: config.tone || defaultChatConfigurationDraft.tone,
    humor_level: clampNumber(config.humorLevel, defaultChatConfigurationDraft.humorLevel, 0, 5),
    system_prompt: config.systemPrompt || "",
    predefined_prompt: config.predefinedPrompt || "",
    generator_deployment_id: config.generatorDeploymentId || defaultChatConfigurationDraft.generatorDeploymentId,
    fallback_deployment_ids: config.fallbackDeploymentIds || [],
    reranker_deployment_id: config.rerankerDeploymentId || "",
    planner_deployment_id: config.plannerDeploymentId || "",
    generation_parameters: config.generationParameters || defaultChatConfigurationDraft.generationParameters,
    citations_enabled: Boolean(config.citations),
    metadata: {
      runtime: "model-farm",
      actual_generator: "deployment",
      route_strategy: config.route || defaultRuntimeRoute(),
      route_mode: answerModeFromRoute(config.route || defaultRuntimeRoute()),
      classifier: config.classifier || "Built-in trained classifier",
      classifier_deployment_id: config.classifierDeploymentId || "",
      planner_deployment_id: config.plannerDeploymentId || "",
      query_embedding_deployment_id: config.queryEmbeddingDeploymentId || "",
      retrieval_mode: retrievalModeValue(config.retrievalMode),
      retrieval_mode_label: config.retrievalMode || "Hybrid",
      top_k: clampNumber(config.topK, 6, 1, 50),
      reranker_enabled: Boolean(config.reranker),
      configuration_id: normalizeConfigurationCode(config.configurationCode) || createConfigurationCode(),
      welcome_message: config.welcomeMessage || defaultChatConfigurationDraft.welcomeMessage,
      conversation_starters: normalizeConversationStarters(config.conversationStarters),
      generator_deployment_id: config.generatorDeploymentId || defaultChatConfigurationDraft.generatorDeploymentId
    }
  };
}

function isValidChatConfigurationDraft(config) {
  return Boolean(
    (config.chatConfigurationId || config.configurationName?.trim()) &&
    config.generatorDeploymentId &&
    config.generatorProvider &&
    config.generatorModel &&
    config.responseStructure &&
    config.tone
  );
}
function formatShortDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const now = new Date();
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

function loadMainLayout() {
  try {
    const raw = JSON.parse(window.localStorage.getItem(MAIN_LAYOUT_STORAGE_KEY) || "{}");
    return {
      historyWidth: clampNumber(raw.historyWidth, DEFAULT_MAIN_LAYOUT.historyWidth, MAIN_LAYOUT_LIMITS.history.min, MAIN_LAYOUT_LIMITS.history.max),
      configWidth: clampNumber(raw.configWidth, DEFAULT_MAIN_LAYOUT.configWidth, MAIN_LAYOUT_LIMITS.config.min, MAIN_LAYOUT_LIMITS.config.max),
      historyCollapsed: Boolean(raw.historyCollapsed),
      configCollapsed: Boolean(raw.configCollapsed)
    };
  } catch {
    return DEFAULT_MAIN_LAYOUT;
  }
}

function saveMainLayout(layout) {
  try {
    window.localStorage.setItem(MAIN_LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // Storage can be unavailable in private or hardened browser modes.
  }
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

function modelDeploymentLabel(deployment = {}) {
  const status = deployment.enabled ? deployment.health_status || "enabled" : "disabled";
  const capabilityText = Array.isArray(deployment.capabilities) ? deployment.capabilities.join(", ") : "";
  return `${deployment.name || deployment.id} - ${deployment.provider || "provider"} / ${deployment.model || "model"} (${status}${capabilityText ? `, ${capabilityText}` : ""})`;
}

function shortDeploymentLabel(deployment = {}) {
  const name = deployment.name || deployment.id || "Deployment";
  const model = deployment.model || "model";
  return truncateMiddle(`${name} - ${model}`, 58);
}

function deploymentOption(deployment = {}) {
  return {
    value: deployment.id,
    label: shortDeploymentLabel(deployment),
    title: modelDeploymentLabel(deployment)
  };
}

function truncateMiddle(value, maxLength = 58) {
  const text = String(value || "");
  if (text.length <= maxLength) return text;
  const keep = Math.max(Math.floor((maxLength - 3) / 2), 8);
  return `${text.slice(0, keep)}...${text.slice(-keep)}`;
}

function uniqueDeploymentName(baseName, deployments = []) {
  const normalizedBaseName = String(baseName || "Model deployment").trim().replace(/\s+/g, " ") || "Model deployment";
  const existingNames = new Set((deployments || []).map((deployment) => String(deployment.name || "").trim().toLowerCase()));
  if (!existingNames.has(normalizedBaseName.toLowerCase())) return normalizedBaseName;
  for (let index = 2; index < 1000; index += 1) {
    const candidate = `${normalizedBaseName} ${index}`;
    if (!existingNames.has(candidate.toLowerCase())) return candidate;
  }
  return `${normalizedBaseName} ${Date.now()}`;
}

function uniqueConnectionName(baseName, connections = []) {
  const normalized = String(baseName || "Model connection").trim() || "Model connection";
  const existingNames = new Set(connections.map((connection) => String(connection.name || "").trim().toLowerCase()));
  if (!existingNames.has(normalized.toLowerCase())) return normalized;
  for (let index = 2; index < 1000; index += 1) {
    const candidate = `${normalized} ${index}`;
    if (!existingNames.has(candidate.toLowerCase())) return candidate;
  }
  return `${normalized} ${Date.now()}`;
}

function connectionProviderForTemplate(template = {}) {
  const provider = String(template.provider || "").trim().toLowerCase();
  if (provider === "local") return "local_builtin";
  if (provider === "google") return "gemini";
  return provider;
}

function emptyModelFarmWizard() {
  return {
    connectionId: "",
    name: "",
    description: "",
    model: "",
    capabilities: [],
    apiBase: "",
    credentialEnvRefs: {},
    credentialSecrets: {},
    defaultParameters: {},
    limits: {},
    pricing: {},
    monthlyBudgetUsd: 0,
    hardBudget: true,
    temperature: 0.2,
    maxTokens: 800,
    timeoutSeconds: 60,
    dimension: 0,
    inputPrice: 0,
    outputPrice: 0,
    metadata: {}
  };
}

function emptyEndpointDraft(defaults = {}) {
  return {
    connectionId: defaults.connection_id || "",
    name: "",
    modelId: defaults.model || "",
    url: defaults.api_base || "",
    apiKey: ""
  };
}

function modelFarmWizardFromDeployment(deployment = {}) {
  const defaultParameters = objectOrEmpty(deployment.default_parameters);
  const limits = objectOrEmpty(deployment.limits);
  const pricing = objectOrEmpty(deployment.pricing);
  const metadata = objectOrEmpty(deployment.metadata);
  return {
    ...emptyModelFarmWizard(),
    connectionId: deployment.connection_id || "",
    name: deployment.name || "",
    description: metadata.description || "",
    model: deployment.model || "",
    capabilities: [...(deployment.capabilities || [])],
    apiBase: deployment.api_base || "",
    credentialEnvRefs: cleanCredentialRefs(deployment.credential_env_refs || {}),
    credentialSecrets: {},
    defaultParameters,
    limits,
    pricing,
    monthlyBudgetUsd: Number(deployment.monthly_budget_usd || 0),
    hardBudget: deployment.hard_budget !== false,
    temperature: Number(defaultParameters.temperature ?? 0.2),
    maxTokens: Number(defaultParameters.max_tokens ?? limits.max_output_tokens ?? 800),
    timeoutSeconds: Number(limits.timeout_seconds || 60),
    dimension: Number(limits.dimension || 0),
    inputPrice: Number(pricing.input_per_million_tokens_usd || 0),
    outputPrice: Number(pricing.output_per_million_tokens_usd || 0),
    metadata
  };
}

function endpointDraftFromDeployment(deployment = {}) {
  return {
    connectionId: deployment.connection_id || "",
    name: deployment.connection_name || deployment.metadata?.endpoint_name || deployment.name || "",
    modelId: deployment.model || "",
    url: deployment.api_base || "",
    apiKey: ""
  };
}

function modelFarmDeploymentPatchPayload(wizard) {
  const payload = modelFarmWizardPayload("", wizard);
  delete payload.template_id;
  delete payload.model;
  delete payload.capabilities;
  delete payload.connection_id;
  delete payload.api_base;
  delete payload.credential_env_refs;
  delete payload.credential_secrets;
  return payload;
}

function templateIdForDeployment(deployment = {}, templates = []) {
  const explicit = deployment.metadata?.template_id || "";
  if (explicit && templates.some((template) => template.id === explicit)) return explicit;
  const provider = String(deployment.provider || "").toLowerCase();
  if (provider === "openrouter") return "openrouter-generation";
  if (provider === "gemini" || provider === "google") return "gemini-generation";
  if (provider === "ollama" || provider === "ollama_chat") return "ollama-generation";
  if (provider === "vllm" || provider === "hosted_vllm") return "vllm-generation";
  const match = templates.find((template) => String(template.provider || "").toLowerCase() === provider);
  return match?.id || explicit || provider || "openai-generation";
}

function templateFromDeployment(deployment = {}) {
  return {
    id: templateIdForDeployment(deployment),
    label: deployment.name || providerLabelFromDeployment(deployment),
    provider_label: providerLabelFromDeployment(deployment),
    provider: deployment.provider || "",
    access_path: deployment.access_path || (deployment.locality === "local" ? "local" : "production"),
    model: deployment.model || "",
    capabilities: deployment.capabilities || [],
    locality: deployment.locality || "remote",
    creatable: deployment.locality !== "local",
    credential_fields: deployment.credential_status?.stored_secret_keys?.includes("api_key") || deployment.credential_status?.references?.length
      ? ["api_key"]
      : [],
    deployment_defaults: {
      name: deployment.name || "",
      model: deployment.model || "",
      api_base: deployment.api_base || "",
      capabilities: deployment.capabilities || [],
      default_parameters: deployment.default_parameters || {},
      limits: deployment.limits || {},
      pricing: deployment.pricing || {},
      monthly_budget_usd: deployment.monthly_budget_usd || 0,
      hard_budget: deployment.hard_budget,
      metadata: deployment.metadata || {}
    }
  };
}

function modelFarmWizardPayload(templateId, wizard) {
  const defaultParameters = { ...(wizard.defaultParameters || {}) };
  const limits = { ...(wizard.limits || {}) };
  const pricing = { ...(wizard.pricing || {}) };
  if (wizard.capabilities.includes("generation") || wizard.capabilities.includes("judge") || wizard.capabilities.includes("planner")) {
    defaultParameters.temperature = Number(wizard.temperature || 0);
    defaultParameters.max_tokens = Number(wizard.maxTokens || 0);
    limits.max_output_tokens = Number(wizard.maxTokens || 0);
  }
  if (Number(wizard.timeoutSeconds || 0) > 0) limits.timeout_seconds = Number(wizard.timeoutSeconds);
  if (wizard.capabilities.includes("embedding") && Number(wizard.dimension || 0) > 0) {
    limits.dimension = Number(wizard.dimension);
  }
  if (Number(wizard.inputPrice || 0) > 0) pricing.input_per_million_tokens_usd = Number(wizard.inputPrice);
  if (Number(wizard.outputPrice || 0) > 0) pricing.output_per_million_tokens_usd = Number(wizard.outputPrice);
  return {
    template_id: templateId,
    connection_id: wizard.connectionId || "",
    name: wizard.name,
    model: wizard.model,
    api_base: wizard.apiBase || "",
    capabilities: wizard.capabilities || [],
    credential_env_refs: cleanCredentialRefs(wizard.credentialEnvRefs || {}),
    credential_secrets: cleanCredentialSecrets(wizard.credentialSecrets || {}),
    default_parameters: defaultParameters,
    limits,
    pricing,
    monthly_budget_usd: Number(wizard.monthlyBudgetUsd || 0),
    hard_budget: Boolean(wizard.hardBudget),
    enabled: false,
    metadata: wizard.metadata || {}
  };
}

function providerTemplatesForTab(templates, tab, query) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  const filtered = templates.filter((template) => {
    const isLocal = template.locality === "local" || !template.creatable;
    if (tab === "local" && !isLocal) return false;
    if (tab !== "local" && isLocal) return false;
    const searchable = [
      template.label,
      template.provider_label,
      template.provider,
      template.model
    ].join(" ").toLowerCase();
    return !normalizedQuery || searchable.includes(normalizedQuery);
  });
  if (tab === "local") return filtered;
  const seen = new Set();
  return filtered.filter((template) => {
    const key = (template.provider_label || template.provider || template.label || template.id).toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function ProviderLogo({ item, className = "" }) {
  const source = providerLogoSource(item);
  const label = providerLabelFromDeployment(item);
  const classes = [
    "ai-provider-logo",
    providerLogoClass(item),
    className,
    source ? "has-image" : ""
  ].filter(Boolean).join(" ");
  return (
    <span className={classes} role="img" aria-label={`${label} logo`} title={label}>
      {source ? <img src={source} alt="" aria-hidden="true" /> : providerLogoText(item)}
    </span>
  );
}

function providerLogoSource(item) {
  if (item?.locality === "local" || providerLabelFromDeployment(item).toLowerCase().includes("local")) {
    return "";
  }
  const provider = String(item?.provider || "").toLowerCase();
  const templateId = String(item?.metadata?.template_id || item?.id || "").toLowerCase();
  const model = String(item?.model || "").toLowerCase();
  if (templateId === "openrouter-generation" || provider === "openrouter") return `${AI_LOGOS_PATH}openrouter.svg`;
  if (templateId === "openai-compatible" || provider === "custom") return `${AI_LOGOS_PATH}openrouter.svg`;
  if (provider === "openai") return `${AI_LOGOS_PATH}openai.svg`;
  if (provider === "azure") return `${AI_LOGOS_PATH}azure-openai.svg`;
  if (provider === "bedrock") return `${AI_LOGOS_PATH}${model.includes("nova") ? "amazon nova-color.svg" : "bedrock-color.svg"}`;
  if (provider === "cohere") return `${AI_LOGOS_PATH}cohere-color.svg`;
  if (provider === "huggingface") return `${AI_LOGOS_PATH}hugging-face.svg`;
  if (provider === "mistral") return `${AI_LOGOS_PATH}mistral-color.svg`;
  if (provider === "gemini") return `${AI_LOGOS_PATH}google-gemini.svg`;
  if (provider === "anthropic") return `${AI_LOGOS_PATH}anthropic.svg`;
  if (provider === "databricks") return `${AI_LOGOS_PATH}databricks.svg`;
  if (provider === "ai21") return `${AI_LOGOS_PATH}ai21.svg`;
  if (provider === "watsonx") return `${AI_LOGOS_PATH}IBM_watsonx_logo.svg`;
  return "";
}

function providerLogoText(item) {
  const label = providerLabelFromDeployment(item);
  const normalized = label.toLowerCase();
  if (normalized.includes("openai")) return "AI";
  if (normalized.includes("azure")) return "AZ";
  if (normalized.includes("bedrock") || normalized.includes("aws")) return "AWS";
  if (normalized.includes("cohere")) return "C";
  if (normalized.includes("hugging")) return "HF";
  if (normalized.includes("mistral")) return "M";
  if (normalized.includes("gemini") || normalized.includes("google")) return "G";
  if (normalized.includes("anthropic")) return "A";
  if (normalized.includes("local")) return "L";
  return label.slice(0, 2).toUpperCase();
}

function providerLogoClass(item) {
  return providerLabelFromDeployment(item).toLowerCase().replace(/[^a-z0-9]+/g, "-");
}

function providerLabelFromDeployment(item) {
  return item?.provider_label || item?.metadata?.provider_label || item?.provider || item?.label || "AI";
}

function requiresEndpointUrl(template) {
  const provider = String(template?.provider || "").toLowerCase();
  return provider === "custom" || provider.includes("azure") || provider.includes("databricks") || provider.includes("watsonx");
}

function chatHistoryDisplayTitle(value, maxLength = 30) {
  const normalized = String(value || "New chat").trim().replace(/\s+/g, " ") || "New chat";
  if (normalized.length <= maxLength) return normalized;
  return `${normalized.slice(0, maxLength - 3).trimEnd()}...`;
}

function cleanCredentialSecrets(secrets = {}) {
  const cleaned = {};
  Object.entries(secrets || {}).forEach(([key, value]) => {
    const secret = String(value || "").trim();
    if (secret) cleaned[key] = secret;
  });
  return cleaned;
}

function cleanCredentialRefs(refs = {}) {
  const cleaned = {};
  Object.entries(refs || {}).forEach(([key, value]) => {
    const envName = String(value || "").trim();
    if (envName) cleaned[key] = envName;
  });
  return cleaned;
}

function objectOrEmpty(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
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
  const provider = raw.embedding_provider || defaultKnowledgeConfiguration.embedding_provider;
  const embeddingModel = raw.embedding_model || defaultKnowledgeConfiguration.embedding_model;
  return {
    chunking_strategy: strategy,
    chunk_size: chunkSize,
    chunk_overlap: chunkingStrategyUsesOverlap(strategy)
      ? clampNumber(raw.chunk_overlap, 120, 0, Math.max(chunkSize - 1, 0))
      : 0,
    embedding_deployment_id: raw.embedding_deployment_id || defaultKnowledgeConfiguration.embedding_deployment_id,
    external_processing_allowed: Boolean(raw.external_processing_allowed),
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
