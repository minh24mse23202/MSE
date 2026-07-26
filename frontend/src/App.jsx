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
  Download,
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
  Square,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  UserPlus,
  X
} from "lucide-react";
import {
  askQuestion,
  askQuestionStream,
  cancelAnswerRequest,
  createChatConfiguration,
  createEvaluationExperiment,
  cancelEvaluationExperiment,
  createKnowledgeBase,
  createModelConnection,
  createModelDeploymentFromTemplate,
  deleteChatConfiguration,
  deleteChatConversation,
  deleteEvaluationExperiment,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  deleteModelDeployment,
  getChatConfigurationLimits,
  getChatMessageTrace,
  getTrace,
  getModelUsageSummary,
  getCurrentUser,
  getKnowledgeProcessingTrace,
  getRagxplainViewerUrl,
  getEvaluationExperiment,
  getEvaluationRun,
  downloadTrace,
  ingestWebsiteSource,
  listChatConfigurations,
  listChatConversations,
  listChatMessageVersions,
  listChatMessages,
  listEvaluationCases,
  listEvaluationDatasets,
  listEvaluationExperiments,
  listEvaluationExperimentRuns,
  listKnowledgeChunks,
  listKnowledgeDocuments,
  listKnowledgeBases,
  listKnowledgeIndexVersions,
  listAgentTools,
  listModelDeployments,
  listModelConnections,
  listModelProviders,
  listModelUsage,
  clearAuthToken,
  hasAuthToken,
  login as loginUser,
  regenerateAnswerStream,
  retryAnswerStream,
  resumeEvaluationExperiment,
  reindexKnowledgeBase,
  submitFeedback,
  startRagxplainDiagnosis,
  testModelDeployment,
  testModelDeploymentDraft,
  testModelConnection,
  listConnectionModels,
  signup as signupUser,
  updateChatConfiguration,
  updateChatConversation,
  updateCurrentUser,
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
  { value: "L3 Complex RAG", label: "L3 Complex RAG" },
  { value: "L4 Advanced RAG", label: "L4 Advanced RAG" }
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
const defaultConversationLimits = {
  defaultCompletedExchanges: 3,
  defaultCharacters: 4000,
  maxCompletedExchanges: 6,
  maxCharacters: 10000
};
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
  conversationAwarenessEnabled: true,
  conversationHistoryExchanges: defaultConversationLimits.defaultCompletedExchanges,
  conversationHistoryCharacters: defaultConversationLimits.defaultCharacters,
  classifierConfidenceThreshold: 0.6,
  classifierMarginThreshold: 0.15,
  agentMaxIterations: 5,
  agentMaxToolCalls: 8,
  agentTimeoutSeconds: 90,
  agentPublicWebEnabled: false,
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
  const [isProfileOpen, setIsProfileOpen] = useState(false);
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
    setIsProfileOpen(false);
    setCurrentUser(null);
    setSignedIn(false);
    setScreen("login");
  }

  async function saveProfile(payload) {
    const result = await updateCurrentUser(payload);
    setCurrentUser(result.user);
    return result.user;
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
    <Shell
      activeScreen={screen}
      onNavigate={setScreen}
      user={currentUser}
      onOpenProfile={() => setIsProfileOpen(true)}
      onSignOut={leaveStudio}
    >
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
      {isProfileOpen && (
        <UserProfileModal
          user={currentUser}
          onClose={() => setIsProfileOpen(false)}
          onSave={saveProfile}
        />
      )}
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

function Shell({ activeScreen, onNavigate, user, onOpenProfile, onSignOut, children }) {
  const userLabel = [user?.first_name, user?.last_name].filter(Boolean).join(" ") || user?.email || "User";
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
          <button
            className="user-chip"
            type="button"
            title="Open user profile"
            aria-label={`Open user profile for ${userLabel}`}
            onClick={onOpenProfile}
          >
            <IconLabel icon={CircleUserRound}>{userLabel}</IconLabel>
          </button>
          <button type="button" aria-label="Sign out" title="Sign out" onClick={onSignOut}>
            <IconOnly icon={LogOut} size={18} />
          </button>
        </div>
      </header>
      <main className="workspace">{children}</main>
    </div>
  );
}

function UserProfileModal({ user, onClose, onSave }) {
  const [form, setForm] = useState({
    firstName: user?.first_name || "",
    lastName: user?.last_name || "",
    email: user?.email || "",
    currentPassword: "",
    newPassword: "",
    confirmPassword: ""
  });
  const [showPasswords, setShowPasswords] = useState(false);
  const [status, setStatus] = useState({ type: "", message: "" });
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape" && !submitting) onClose();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose, submitting]);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setStatus({ type: "", message: "" });
  }

  async function submitProfile(event) {
    event.preventDefault();
    const normalizedEmail = form.email.trim().toLowerCase();
    const emailChanged = normalizedEmail !== String(user?.email || "").toLowerCase();
    const passwordChanged = Boolean(form.newPassword);
    if (passwordChanged && form.newPassword !== form.confirmPassword) {
      setStatus({ type: "error", message: "New password and confirmation do not match." });
      return;
    }
    if ((emailChanged || passwordChanged) && !form.currentPassword) {
      setStatus({ type: "error", message: "Enter your current password to change email or password." });
      return;
    }

    setSubmitting(true);
    setStatus({ type: "", message: "" });
    try {
      const payload = {
        first_name: form.firstName.trim(),
        last_name: form.lastName.trim(),
        email: normalizedEmail,
        current_password: form.currentPassword
      };
      if (passwordChanged) payload.new_password = form.newPassword;
      const updated = await onSave(payload);
      setForm((current) => ({
        ...current,
        firstName: updated.first_name || "",
        lastName: updated.last_name || "",
        email: updated.email || "",
        currentPassword: "",
        newPassword: "",
        confirmPassword: ""
      }));
      setStatus({ type: "success", message: "Profile updated successfully." });
    } catch (error) {
      setStatus({ type: "error", message: error.message });
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop profile-modal-backdrop" role="presentation" onMouseDown={submitting ? undefined : onClose}>
      <section
        className="modal profile-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="profile-modal-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <h2 id="profile-modal-title"><IconLabel icon={CircleUserRound} size={20}>User profile</IconLabel></h2>
            <p>Update your account information and login credentials.</p>
          </div>
          <button className="icon-button modal-close" type="button" aria-label="Close profile" onClick={onClose} disabled={submitting}>
            <IconOnly icon={X} />
          </button>
        </header>

        <form className="profile-form" onSubmit={submitProfile}>
          <div className="profile-identity">
            <span><strong>Account ID</strong>{user?.id || "Unavailable"}</span>
            <span><strong>Role</strong>{user?.role || "user"}</span>
          </div>

          <section className="profile-section">
            <div>
              <h3>Personal information</h3>
              <p>Your display name is shown in the application header.</p>
            </div>
            <div className="profile-form-grid">
              <label>
                First name
                <input
                  value={form.firstName}
                  onChange={(event) => updateField("firstName", event.target.value)}
                  autoComplete="given-name"
                  maxLength={100}
                />
              </label>
              <label>
                Last name
                <input
                  value={form.lastName}
                  onChange={(event) => updateField("lastName", event.target.value)}
                  autoComplete="family-name"
                  maxLength={100}
                />
              </label>
            </div>
            <label>
              Email
              <input
                type="email"
                value={form.email}
                onChange={(event) => updateField("email", event.target.value)}
                autoComplete="email"
                required
              />
            </label>
          </section>

          <section className="profile-section">
            <div>
              <h3>Password</h3>
              <p>Current password is required when changing your email or password.</p>
            </div>
            <label>
              Current password
              <div className="profile-password-input">
                <input
                  type={showPasswords ? "text" : "password"}
                  value={form.currentPassword}
                  onChange={(event) => updateField("currentPassword", event.target.value)}
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  aria-label={showPasswords ? "Hide passwords" : "Show passwords"}
                  title={showPasswords ? "Hide passwords" : "Show passwords"}
                  onClick={() => setShowPasswords((current) => !current)}
                >
                  <IconOnly icon={showPasswords ? EyeOff : Eye} size={17} />
                </button>
              </div>
            </label>
            <div className="profile-form-grid">
              <label>
                New password
                <input
                  type={showPasswords ? "text" : "password"}
                  value={form.newPassword}
                  onChange={(event) => updateField("newPassword", event.target.value)}
                  autoComplete="new-password"
                  minLength={8}
                  placeholder="Leave blank to keep current"
                />
              </label>
              <label>
                Confirm new password
                <input
                  type={showPasswords ? "text" : "password"}
                  value={form.confirmPassword}
                  onChange={(event) => updateField("confirmPassword", event.target.value)}
                  autoComplete="new-password"
                  minLength={8}
                />
              </label>
            </div>
          </section>

          {status.message && (
            <p className={`profile-status ${status.type}`} role={status.type === "error" ? "alert" : "status"}>
              {status.message}
            </p>
          )}

          <div className="modal-actions profile-actions">
            <button className="secondary-action" type="button" onClick={onClose} disabled={submitting}>Cancel</button>
            <button className="primary-action" type="submit" disabled={submitting}>
              <IconLabel icon={Save}>{submitting ? "Saving..." : "Save profile"}</IconLabel>
            </button>
          </div>
        </form>
      </section>
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
  const activeAnswerOperationRef = useRef(null);
  const [messages, setMessages] = useState(() => welcomeMessagesFromConfig(defaultChatConfigurationDraft));
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
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
  const [conversationLimits, setConversationLimits] = useState(defaultConversationLimits);
  const [configurationStatus, setConfigurationStatus] = useState("");
  const [layout, setLayout] = useState(loadMainLayout);
  const [config, setConfig] = useState({
    classifier: "DistilBERT",
    classifierDeploymentId: "",
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
    getChatConfigurationLimits()
      .then((payload) => {
        const limits = normalizeConversationLimits(payload);
        setConversationLimits(limits);
        setConfig((current) => ({
          ...current,
          conversationHistoryExchanges: clampNumber(
            current.conversationHistoryExchanges,
            limits.defaultCompletedExchanges,
            1,
            limits.maxCompletedExchanges
          ),
          conversationHistoryCharacters: clampNumber(
            current.conversationHistoryCharacters,
            limits.defaultCharacters,
            1,
            limits.maxCharacters
          )
        }));
      })
      .catch((error) => setConfigurationStatus(`Conversation limits unavailable: ${error.message}`));
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
      }, conversationLimits));
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
      const updated = await updateChatConfiguration(
        config.chatConfigurationId,
        chatConfigurationPayloadFromDraft(config, conversationLimits)
      );
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

  function beginAnswerOperation(requestId, controller, messageId = "") {
    activeAnswerOperationRef.current = {
      requestId,
      controller,
      messageId,
      stopping: false,
      abortTimer: null
    };
  }

  function finishAnswerOperation(requestId) {
    const operation = activeAnswerOperationRef.current;
    if (!operation || operation.requestId !== requestId) return;
    if (operation.abortTimer) window.clearTimeout(operation.abortTimer);
    activeAnswerOperationRef.current = null;
    setIsStopping(false);
  }

  async function stopActiveAnswer() {
    const operation = activeAnswerOperationRef.current;
    if (!operation || operation.stopping) return;
    operation.stopping = true;
    setIsStopping(true);
    setFeedbackStatus("Stopping answer...");
    try {
      await cancelAnswerRequest(operation.requestId);
      operation.abortTimer = window.setTimeout(() => operation.controller.abort(), 3000);
    } catch (error) {
      operation.stopping = false;
      setIsStopping(false);
      setFeedbackStatus(`Stop failed: ${error.message}`);
    }
  }

  async function sendQuestion() {
    const trimmed = question.trim();
    if (!trimmed) return;
    const mode = answerModeFromRoute(config.route);
    const requiresKnowledgeBase = mode !== "direct";
    if (requiresKnowledgeBase && !selectedKnowledgeBaseId) {
      setFeedbackStatus("Select a knowledge base before using Adaptive, L2 Simple RAG, L3 Complex RAG, or L4 Advanced RAG.");
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
    const streamController = new AbortController();
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
    beginAnswerOperation(streamRequestId, streamController, assistantMessageId);
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
        chatConfiguration: chatConfigurationPayloadFromDraft(config, conversationLimits),
        signal: streamController.signal
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
        if (event.type === "cancelled") {
          patchStreamingAssistant((message) => ({
            content: event.data?.partial_answer || message.content || "",
            status: "cancelled",
            latestVersionStatus: "cancelled",
            streaming: false,
            streamingStatus: "",
            metadata: {
              ...(message.metadata || {}),
              request_id: event.data?.request_id || streamRequestId,
              assistant_message_id: event.data?.assistant_message_id || persistedAssistantMessageId,
              message_version_id: event.data?.message_version_id || message.metadata?.message_version_id,
              message_version_number: event.data?.message_version_number || message.metadata?.message_version_number,
              cancelled: true
            }
          }));
        }
      });
      if (response?.status === "cancelled") {
        stopStreamPolling();
        setFeedbackStatus("Answer stopped");
        await refreshConversationLists();
        return;
      }
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
        latestVersionStatus: "completed",
        streaming: false,
        streamingStatus: "",
        versionCount: 1,
        latestVersionNumber: Number(response.metadata?.message_version_number || 1),
        viewingVersionNumber: Number(response.metadata?.message_version_number || 1)
      });
      await refreshConversationLists();
    } catch (error) {
      stopStreamPolling();
      const stopped = streamController.signal.aborted || Boolean(activeAnswerOperationRef.current?.stopping);
      patchStreamingAssistant((message) => ({
          question: trimmed,
          role: "assistant",
          content: stopped ? message.content : `Answer request failed: ${error.message}`,
          contexts: stopped ? message.contexts : [],
        metadata: {
          ...(message.metadata || {}),
          error: error.message,
          complexity_label: stopped ? message.metadata?.complexity_label : "unknown",
          trace_steps: stopped ? message.metadata?.trace_steps || [] : [],
          request_id: streamRequestId,
          assistant_message_id: persistedAssistantMessageId,
          cancelled: stopped
        },
        status: stopped ? "cancelled" : "failed",
        latestVersionStatus: stopped ? "cancelled" : "failed",
        streaming: false,
        streamingStatus: ""
      }));
      setFeedbackStatus(stopped ? "Answer stopped" : `Answer failed: ${error.message}`);
    } finally {
      stopStreamPolling();
      finishAnswerOperation(streamRequestId);
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

  async function copyAssistantMessage(message) {
    try {
      await navigator.clipboard.writeText(message.content || "");
      setFeedbackStatus("Answer copied");
    } catch {
      setFeedbackStatus("Copy failed: clipboard access is unavailable.");
    }
  }

  async function navigateAnswerVersion(message, direction) {
    const persistedMessageId = message.metadata?.assistant_message_id || message.id;
    try {
      const versions = message.answerVersions || await listChatMessageVersions(persistedMessageId);
      if (!versions.length) return;
      const canonicalNumber = Number(message.metadata?.message_version_number || 1);
      const currentNumber = Number(message.viewingVersionNumber || canonicalNumber);
      const foundIndex = versions.findIndex((version) => version.version_number === currentNumber);
      const currentIndex = foundIndex >= 0 ? foundIndex : versions.length - 1;
      const nextIndex = Math.max(0, Math.min(versions.length - 1, currentIndex + direction));
      const selected = versions[nextIndex];
      setMessages((current) => updateMessage(current, message.id, {
        content: selected.content,
        contexts: selected.contexts || [],
        metadata: {
          ...(selected.metadata || {}),
          assistant_message_id: persistedMessageId,
          message_version_id: selected.id,
          message_version_number: selected.version_number
        },
        status: selected.status,
        streaming: false,
        answerVersions: versions,
        versionCount: versions.length,
        latestVersionNumber: Math.max(...versions.map((version) => Number(version.version_number || 0))),
        viewingVersionNumber: selected.version_number
      }));
    } catch (error) {
      setFeedbackStatus(`Answer versions unavailable: ${error.message}`);
    }
  }

  async function runExistingAnswer(message, operation = "regenerate") {
    if (isLoading) return;
    const isRetry = operation === "retry";
    const actionLabel = isRetry ? "retrying" : "regenerating";
    const mode = answerModeFromRoute(config.route);
    if (mode !== "direct" && !selectedKnowledgeBaseId) {
      setFeedbackStatus(`Select a knowledge base before ${actionLabel} this answer.`);
      return;
    }
    if (!isValidChatConfigurationDraft(config)) {
      setFeedbackStatus(`Select or save a chatbot configuration before ${actionLabel}.`);
      return;
    }
    const selectedKnowledgeConfiguration = selectedKnowledgeBase
      ? knowledgeConfigurationFromRecord(selectedKnowledgeBase)
      : {};
    const configuredDeploymentIds = [
      config.generatorDeploymentId,
      ...(config.fallbackDeploymentIds || []),
      config.plannerDeploymentId,
      config.classifierDeploymentId,
      config.reranker ? config.rerankerDeploymentId : ""
    ].filter(Boolean);
    const remoteDeployment = configuredDeploymentIds
      .map((deploymentId) => modelDeployments.find((deployment) => deployment.id === deploymentId))
      .find((deployment) => deployment && deployment.locality !== "local");
    if (selectedKnowledgeBase && remoteDeployment && !selectedKnowledgeConfiguration.external_processing_allowed) {
      setFeedbackStatus(
        `Remote model "${remoteDeployment.name}" is blocked for this knowledge base. `
        + "Enable Allow remote model processing in Knowledge Bases first."
      );
      return;
    }

    const persistedMessageId = message.metadata?.assistant_message_id || message.id;
    const requestId = createId();
    const streamController = new AbortController();
    const canonicalSnapshot = {
      content: message.content,
      contexts: message.contexts || [],
      metadata: message.metadata || {},
      status: message.status || "completed",
      viewingVersionNumber: message.viewingVersionNumber
    };
    let nextVersionNumber = Number(message.latestVersionNumber || message.metadata?.message_version_number || 1) + 1;
    let versionCreated = false;
    setIsLoading(true);
    beginAnswerOperation(requestId, streamController, persistedMessageId);
    setMessages((current) => updateMessage(current, message.id, {
      content: "",
      contexts: [],
      metadata: {
        ...(message.metadata || {}),
        request_id: requestId,
        assistant_message_id: persistedMessageId,
        regenerated: !isRetry,
        retried: isRetry,
        trace_steps: []
      },
      status: "streaming",
      streaming: true,
      streamingStatus: isRetry ? "Preparing retry..." : "Preparing regenerated answer..."
    }));
    try {
      const streamFunction = isRetry ? retryAnswerStream : regenerateAnswerStream;
      const response = await streamFunction(persistedMessageId, {
        requestId,
        knowledgeBaseId: selectedKnowledgeBaseId,
        documentIds: selectedFilterDocumentIds,
        mode,
        retrievalMode: retrievalModeValue(config.retrievalMode),
        topK: config.topK,
        chatConfigurationId: config.chatConfigurationId || null,
        chatConfiguration: chatConfigurationPayloadFromDraft(config, conversationLimits),
        signal: streamController.signal
      }, (event) => {
        if (event.type === "started") {
          versionCreated = Boolean(event.data?.message_version_id);
          nextVersionNumber = Number(event.data?.message_version_number || nextVersionNumber);
          setMessages((current) => updateMessage(current, message.id, (currentMessage) => ({
            metadata: {
              ...(currentMessage.metadata || {}),
              request_id: event.data?.request_id || requestId,
              assistant_message_id: persistedMessageId,
              message_version_id: event.data?.message_version_id,
              message_version_number: nextVersionNumber
            },
            streamingStatus: "Route is running..."
          })));
        } else if (event.type === "trace") {
          setMessages((current) => updateMessage(current, message.id, (currentMessage) => ({
            metadata: {
              ...(currentMessage.metadata || {}),
              trace_steps: [...(currentMessage.metadata?.trace_steps || []), event.data]
            },
            streamingStatus: event.data?.detail || "Running Adaptive RAG..."
          })));
        } else if (event.type === "sources") {
          setMessages((current) => updateMessage(current, message.id, {
            contexts: event.data?.contexts || [],
            streamingStatus: "Sources retrieved. Generating answer..."
          }));
        } else if (event.type === "delta") {
          setMessages((current) => updateMessage(current, message.id, (currentMessage) => ({
            content: `${currentMessage.content || ""}${event.data?.text || ""}`,
            streamingStatus: isRetry ? "Streaming retry..." : "Streaming regenerated answer..."
          })));
        } else if (event.type === "cancelled") {
          setMessages((current) => updateMessage(current, message.id, (currentMessage) => ({
            content: event.data?.partial_answer || currentMessage.content || "",
            status: "cancelled",
            latestVersionStatus: "cancelled",
            streaming: false,
            streamingStatus: "",
            metadata: {
              ...(currentMessage.metadata || {}),
              request_id: event.data?.request_id || requestId,
              message_version_id: event.data?.message_version_id || currentMessage.metadata?.message_version_id,
              message_version_number: event.data?.message_version_number || nextVersionNumber,
              cancelled: true
            }
          })));
        }
      });
      if (response?.status === "cancelled") {
        setMessages((current) => updateMessage(current, message.id, {
          status: "cancelled",
          latestVersionStatus: "cancelled",
          streaming: false,
          streamingStatus: "",
          answerVersions: null,
          versionCount: Math.max(Number(message.versionCount || 1) + 1, nextVersionNumber),
          latestVersionNumber: nextVersionNumber,
          viewingVersionNumber: nextVersionNumber
        }));
        await refreshConversationLists();
        setFeedbackStatus("Answer stopped");
        return;
      }
      setMessages((current) => updateMessage(current, message.id, {
        question: response.question,
        content: response.answer,
        contexts: response.contexts || [],
        metadata: {
          ...(response.metadata || {}),
          assistant_message_id: persistedMessageId
        },
        status: "completed",
        latestVersionStatus: "completed",
        streaming: false,
        streamingStatus: "",
        answerVersions: null,
        versionCount: Math.max(Number(message.versionCount || 1) + 1, nextVersionNumber),
        latestVersionNumber: nextVersionNumber,
        viewingVersionNumber: nextVersionNumber
      }));
      await refreshConversationLists();
      setFeedbackStatus(`${isRetry ? "Retried" : "Generated"} answer version ${nextVersionNumber}`);
    } catch (error) {
      const stopped = streamController.signal.aborted || Boolean(activeAnswerOperationRef.current?.stopping);
      setMessages((current) => updateMessage(current, message.id, (currentMessage) => ({
        ...(versionCreated ? {} : canonicalSnapshot),
        status: versionCreated ? (stopped ? "cancelled" : "failed") : canonicalSnapshot.status,
        latestVersionStatus: versionCreated
          ? (stopped ? "cancelled" : "failed")
          : message.latestVersionStatus,
        streaming: false,
        streamingStatus: "",
        answerVersions: null,
        metadata: versionCreated
          ? { ...(currentMessage.metadata || {}), error: error.message, cancelled: stopped }
          : canonicalSnapshot.metadata,
        versionCount: versionCreated
          ? Math.max(Number(message.versionCount || 1) + 1, nextVersionNumber)
          : Number(message.versionCount || 1),
        latestVersionNumber: versionCreated
          ? nextVersionNumber
          : Number(message.latestVersionNumber || message.metadata?.message_version_number || 1),
        viewingVersionNumber: versionCreated ? nextVersionNumber : canonicalSnapshot.viewingVersionNumber
      })));
      setFeedbackStatus(stopped ? "Answer stopped" : `${isRetry ? "Retry" : "Regeneration"} failed: ${error.message}`);
    } finally {
      finishAnswerOperation(requestId);
      setIsLoading(false);
    }
  }

  async function regenerateMessage(message) {
    return runExistingAnswer(message, "regenerate");
  }

  async function retryMessage(message) {
    return runExistingAnswer(message, "retry");
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
        interactionLocked={isLoading}
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
        isStopping={isStopping}
        onSend={sendQuestion}
        onStop={stopActiveAnswer}
        onOpenPopup={setPopup}
        onFeedback={recordFeedback}
        onCopyMessage={copyAssistantMessage}
        onRegenerate={regenerateMessage}
        onRetry={retryMessage}
        onNavigateVersion={navigateAnswerVersion}
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
        conversationLimits={conversationLimits}
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
      {popup && <TraceModal popup={popup} onClose={() => setPopup(null)} onOpenSource={(candidate) => {
        const sourceId = candidate.chunk_id;
        const currentContexts = popup.message.contexts || [];
        const exists = currentContexts.some((context) => context.id === sourceId);
        const diagnosticContext = {
          id: sourceId,
          rank: candidate.selected_rank || candidate.bm25_rank || candidate.dense_rank || 0,
          score: candidate.hybrid_score || candidate.dense_raw_score || candidate.bm25_raw_score || 0,
          mode: candidate.retrieval_step ? "multi-step" : "diagnostic",
          text: candidate.text || "",
          metadata: { ...(candidate.metadata || {}), document_id: candidate.document_id, diagnostic_candidate: true }
        };
        setPopup({
          type: "source",
          message: { ...popup.message, contexts: exists ? currentContexts : [...currentContexts, diagnosticContext] },
          selectedSourceId: sourceId
        });
      }} />}
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
  interactionLocked = false,
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
      <button className="new-chat-action" type="button" onClick={onNewChat} disabled={interactionLocked}><IconLabel icon={MessageSquarePlus}>New chat</IconLabel></button>
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
        interactionLocked={interactionLocked}
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
        interactionLocked={interactionLocked}
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
  onDeleteConversation,
  interactionLocked = false
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
                <button className="conversation-main" type="button" disabled={interactionLocked} onClick={() => onSelectConversation(conversation)}>
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
                    disabled={interactionLocked}
                    aria-label={conversation.pinned ? "Unpin chat" : "Pin chat"}
                    onClick={() => onTogglePinned(conversation)}
                  >
                    <IconOnly icon={conversation.pinned ? PinOff : Pin} size={14} />
                  </button>
                  <button type="button" disabled={interactionLocked} aria-label="Rename chat" onClick={() => beginRename(conversation)}>
                    <IconOnly icon={Pencil} size={14} />
                  </button>
                  <button type="button" disabled={interactionLocked} aria-label="Delete chat" onClick={() => onDeleteConversation(conversation)}>
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
  isStopping,
  onSend,
  onStop,
  onOpenPopup,
  onFeedback,
  onCopyMessage,
  onRegenerate,
  onRetry,
  onNavigateVersion,
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
  const latestCompletedAssistantId = [...messages].reverse().find(
    (message) => message.role === "assistant" && message.status === "completed" && !isWelcomeMessage(message)
  )?.id;
  const latestAssistantId = [...messages].reverse().find(
    (message) => message.role === "assistant" && !isWelcomeMessage(message)
  )?.id;
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
                  <AssistantMessageContent
                    content={message.content}
                    citationSources={message.metadata?.citation_sources || {}}
                    onCitationClick={(label, source) => onOpenPopup({
                      type: "source",
                      message,
                      sourceLabel: label,
                      selectedSourceId: source?.context_id || source?.chunk_id || ""
                    })}
                    onInvalidCitation={() => onOpenPopup({
                      type: "trace",
                      message,
                      focusStep: "Citation validation"
                    })}
                  />
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
                <button type="button" onClick={() => onCopyMessage(message)}><IconLabel icon={Copy}>Copy</IconLabel></button>
                <button onClick={() => onOpenPopup({ type: "trace", message })}><IconLabel icon={GitBranch}>Trace</IconLabel></button>
                {message.id === latestCompletedAssistantId && (
                  <button type="button" disabled={isLoading} onClick={() => onRegenerate(message)}>
                    <IconLabel icon={RotateCw}>Regenerate</IconLabel>
                  </button>
                )}
                {message.id === latestAssistantId && (
                  ["failed", "cancelled"].includes(message.status)
                  || ["failed", "cancelled"].includes(message.latestVersionStatus)
                ) && (
                  <button type="button" disabled={isLoading} onClick={() => onRetry(message)}>
                    <IconLabel icon={RefreshCw}>Retry</IconLabel>
                  </button>
                )}
                <button onClick={() => onFeedback(message, "up")}><IconLabel icon={ThumbsUp}>Useful</IconLabel></button>
                <button onClick={() => onFeedback(message, "down")}><IconLabel icon={ThumbsDown}>Needs work</IconLabel></button>
                {message.metadata?.query_rewritten && (
                  <button
                    className="follow-up-resolved-chip"
                    type="button"
                    title={message.metadata?.standalone_query || "Open query reformulation trace"}
                    onClick={() => onOpenPopup({ type: "trace", message, focusStep: "Query reformulation" })}
                  >
                    <IconLabel icon={RotateCw}>Follow-up resolved</IconLabel>
                  </button>
                )}
                {message.metadata?.citation_validation?.status === "warning" && (
                  <button
                    className="citation-warning-chip"
                    type="button"
                    title={message.metadata.citation_validation.detail || "Open citation validation trace"}
                    onClick={() => onOpenPopup({ type: "trace", message, focusStep: "Citation validation" })}
                  >
                    <IconLabel icon={AlertTriangle}>Citation warning</IconLabel>
                  </button>
                )}
                <span><IconLabel icon={BrainCircuit}>{message.metadata?.complexity_label || "pending"}</IconLabel></span>
                {Number(message.versionCount || message.latestVersionNumber || 1) > 1 && (
                  <span className="answer-version-nav">
                    <button
                      type="button"
                      aria-label="Previous answer version"
                      disabled={Number(message.viewingVersionNumber || message.metadata?.message_version_number || 1) <= 1}
                      onClick={() => onNavigateVersion(message, -1)}
                    >
                      <IconOnly icon={ChevronLeft} size={14} />
                    </button>
                    <em>
                      {Number(message.viewingVersionNumber || message.metadata?.message_version_number || 1)}
                      /{Number(message.latestVersionNumber || message.versionCount || 1)}
                    </em>
                    <button
                      type="button"
                      aria-label="Next answer version"
                      disabled={
                        Number(message.viewingVersionNumber || message.metadata?.message_version_number || 1)
                        >= Number(message.latestVersionNumber || message.versionCount || 1)
                      }
                      onClick={() => onNavigateVersion(message, 1)}
                    >
                      <IconOnly icon={ChevronRight} size={14} />
                    </button>
                  </span>
                )}
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
          disabled={isLoading}
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
            disabled={isLoading || !selectedKnowledgeBaseId}
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
            disabled={isLoading}
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
          <button
            className={`send-action ${isLoading ? "stop-action" : ""}`}
            type="button"
            onClick={isLoading ? onStop : onSend}
            disabled={isLoading ? isStopping : (requiresKnowledgeBase && !selectedKnowledgeBaseId)}
            aria-label={isLoading ? (isStopping ? "Stopping answer" : "Stop answer") : "Send message"}
          >
            {isLoading ? <Square size={17} fill="currentColor" aria-hidden="true" /> : <SendHorizontal size={20} aria-hidden="true" />}
          </button>
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
  conversationLimits = defaultConversationLimits,
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
  const classifierDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("classifier"));
  const rerankerDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("rerank"));
  const plannerDeployments = modelDeployments.filter((deployment) => deployment.capabilities?.includes("planner"));
  const activeKnowledgeConfiguration = selectedKnowledgeBase ? knowledgeConfigurationFromRecord(selectedKnowledgeBase) : {};
  const activeEmbeddingDeploymentId = selectedKnowledgeBase ? activeKnowledgeConfiguration.embedding_deployment_id || "" : "";
  const executedQueryEmbeddingModel = selectedKnowledgeBase
    ? selectedKnowledgeBase.embedding_model || activeKnowledgeConfiguration.embedding_model || "Not indexed"
    : "";
  const activeEmbeddingDeployment = modelDeployments.find((deployment) => deployment.id === activeEmbeddingDeploymentId);
  const activeQueryEmbeddingLabel = config.retrievalMode === "BM25"
    ? "Not used by BM25 retrieval"
    : (
        activeEmbeddingDeployment?.name
        || activeEmbeddingDeployment?.model
        || executedQueryEmbeddingModel
        || "Not indexed"
      );
  const selectedChatConfiguration = chatConfigurations.find((item) => item.id === config.chatConfigurationId);
  const configurationCreatedAt = selectedChatConfiguration?.created_at || config.configurationCreatedAt || "";
  const configurationUpdatedAt = selectedChatConfiguration?.updated_at || config.configurationUpdatedAt || "";
  const conversationContextEnabled = Boolean(config.conversationAwarenessEnabled);
  const [collapsedSections, setCollapsedSections] = useState({ knowledge: true });
  const [agentTools, setAgentTools] = useState([]);
  const [agentToolsStatus, setAgentToolsStatus] = useState("");
  useEffect(() => {
    let active = true;
    listAgentTools(Boolean(config.agentPublicWebEnabled))
      .then((tools) => {
        if (active) {
          setAgentTools(tools);
          setAgentToolsStatus("");
        }
      })
      .catch((error) => {
        if (active) {
          setAgentTools([]);
          setAgentToolsStatus(error.message);
        }
      });
    return () => {
      active = false;
    };
  }, [config.agentPublicWebEnabled]);
  function toggleCustomizerSection(sectionId) {
    setCollapsedSections((current) => ({ ...current, [sectionId]: !current[sectionId] }));
  }
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
        <label className="check-row">
          <input
            type="checkbox"
            checked={conversationContextEnabled}
            onChange={(event) => setConfig({ ...config, conversationAwarenessEnabled: event.target.checked })}
          />
          Use conversation context for follow-up questions
        </label>
        <div
          className={`config-two-column conversation-memory-controls ${conversationContextEnabled ? "" : "is-disabled"}`}
          aria-disabled={!conversationContextEnabled}
        >
          <label>
            Completed exchanges
            <input
              type="number"
              min="1"
              max={conversationLimits.maxCompletedExchanges}
              disabled={!conversationContextEnabled}
              value={config.conversationHistoryExchanges}
              onChange={(event) => setConfig({
                ...config,
                conversationHistoryExchanges: clampNumber(
                  event.target.value,
                  conversationLimits.defaultCompletedExchanges,
                  1,
                  conversationLimits.maxCompletedExchanges
                )
              })}
            />
          </label>
          <label>
            Character budget
            <input
              type="number"
              min="1"
              max={conversationLimits.maxCharacters}
              disabled={!conversationContextEnabled}
              value={config.conversationHistoryCharacters}
              onChange={(event) => setConfig({
                ...config,
                conversationHistoryCharacters: clampNumber(
                  event.target.value,
                  conversationLimits.defaultCharacters,
                  1,
                  conversationLimits.maxCharacters
                )
              })}
            />
          </label>
        </div>
        <small>
          Server maximum: {conversationLimits.maxCompletedExchanges} completed exchanges and{" "}
          {conversationLimits.maxCharacters.toLocaleString()} characters. Follow-up rewriting appears in Trace.
        </small>
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
              ...classifierDeployments.map((deployment) => classifierDeploymentOption(deployment))
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
              { value: "", label: "No planner (L4 falls back to L3)" },
              ...plannerDeployments.map((deployment) => deploymentOption(deployment))
            ]}
            onChange={(plannerDeploymentId) => setConfig({ ...config, plannerDeploymentId })}
          />
        </div>
        <div className="config-two-column">
          <label>
            Confidence threshold
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={config.classifierConfidenceThreshold}
              onChange={(event) => setConfig({
                ...config,
                classifierConfidenceThreshold: clampNumber(event.target.value, 0.6, 0, 1)
              })}
            />
            <small>Adaptive routing escalates one level below this confidence.</small>
          </label>
          <label>
            Top-two margin threshold
            <input
              type="number"
              min="0"
              max="1"
              step="0.05"
              value={config.classifierMarginThreshold}
              onChange={(event) => setConfig({
                ...config,
                classifierMarginThreshold: clampNumber(event.target.value, 0.15, 0, 1)
              })}
            />
            <small>Adaptive routing escalates when the two leading classes are too close.</small>
          </label>
        </div>
        <label>
          Query embedding
          <input
            value={selectedKnowledgeBase ? activeQueryEmbeddingLabel : "Select a knowledge base first"}
            readOnly
            aria-readonly="true"
            title={
              selectedKnowledgeBase
                ? `Inherited from active Knowledge Base index${activeEmbeddingDeploymentId ? ` (${activeEmbeddingDeploymentId})` : ""}`
                : "Select a knowledge base first"
            }
          />
          <small>
            {config.retrievalMode === "BM25"
              ? "BM25 is lexical and does not generate a query vector."
              : "Locked to the active Knowledge Base index to preserve vector-space compatibility."}
          </small>
        </label>
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
        {(config.route === "Adaptive" || config.route === "L4 Advanced RAG") && (
          <div className="agent-runtime-settings">
            <h4>L4 agent limits</h4>
            <div className="config-three-column">
              <label>
                Iterations
                <input
                  type="number"
                  min="1"
                  max="8"
                  value={config.agentMaxIterations}
                  onChange={(event) => setConfig({ ...config, agentMaxIterations: clampNumber(event.target.value, 5, 1, 8) })}
                />
              </label>
              <label>
                Tool calls
                <input
                  type="number"
                  min="1"
                  max="12"
                  value={config.agentMaxToolCalls}
                  onChange={(event) => setConfig({ ...config, agentMaxToolCalls: clampNumber(event.target.value, 8, 1, 12) })}
                />
              </label>
              <label>
                Timeout (seconds)
                <input
                  type="number"
                  min="30"
                  max="180"
                  value={config.agentTimeoutSeconds}
                  onChange={(event) => setConfig({ ...config, agentTimeoutSeconds: clampNumber(event.target.value, 90, 30, 180) })}
                />
              </label>
            </div>
            <label className="check-row">
              <input
                type="checkbox"
                checked={Boolean(config.agentPublicWebEnabled)}
                onChange={(event) => setConfig({ ...config, agentPublicWebEnabled: event.target.checked })}
              />
              Allow request-scoped public website fetching
            </label>
            <div className="agent-tool-availability" aria-label="L4 agent tool availability">
              {agentTools.map((tool) => (
                <span
                  key={tool.name}
                  className={tool.available ? "is-available" : "is-unavailable"}
                  title={tool.available ? tool.description : tool.unavailable_reason}
                >
                  {tool.name.replaceAll("_", " ")} · {tool.available ? "available" : "unavailable"}
                </span>
              ))}
            </div>
            {agentToolsStatus && <p className="config-status-note">Tool registry unavailable: {agentToolsStatus}</p>}
            {!config.plannerDeploymentId && config.route === "L4 Advanced RAG" && (
              <p className="config-status-note warning">Select an enabled planner model. Without one, L4 safely falls back to L3.</p>
            )}
          </div>
        )}
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
function TraceModal({ popup, onClose, onOpenSource }) {
  const { type, message, focusStep = "", selectedSourceId = "", sourceLabel = "" } = popup;
  const contexts = message.contexts || [];
  const legacyTraceSteps = Array.isArray(message.metadata?.trace_steps) ? message.metadata.trace_steps : [];
  const [sourceQuery, setSourceQuery] = useState("");
  const [activeSourceId, setActiveSourceId] = useState(selectedSourceId || contexts[0]?.id || "");
  const [traceQuery, setTraceQuery] = useState("");
  const [selectedTraceIndex, setSelectedTraceIndex] = useState(0);
  const [traceReport, setTraceReport] = useState(null);
  const [traceLoading, setTraceLoading] = useState(false);
  const [traceError, setTraceError] = useState("");
  const [traceTab, setTraceTab] = useState("inputs");
  const [traceOverviewCollapsed, setTraceOverviewCollapsed] = useState(true);
  const durableSpans = Array.isArray(traceReport?.spans) ? traceReport.spans : [];
  const executionSpans = durableSpans.filter(
    (span) => !(span.category === "orchestration" && !span.parent_span_id)
  );
  const traceSteps = durableSpans.length > 0
    ? executionSpans.map((span) => ({
        step: span.name,
        status: span.status,
        detail: span.detail,
        metadata: { ...(span.metrics || {}), duration_ms: span.duration_ms, category: span.category },
        span
      }))
    : legacyTraceSteps;

  useEffect(() => {
    setSourceQuery("");
    setActiveSourceId(selectedSourceId || contexts[0]?.id || "");
    setTraceQuery("");
    setTraceTab("inputs");
    setTraceOverviewCollapsed(true);
    const focusedIndex = focusStep
      ? traceSteps.findIndex((step) => step.step === focusStep)
      : -1;
    setSelectedTraceIndex(focusedIndex >= 0 ? focusedIndex : 0);
  }, [message.id, type, focusStep, selectedSourceId, traceReport?.trace_id]);

  useEffect(() => {
    let active = true;
    if (type !== "trace") return () => { active = false; };
    setTraceReport(null);
    setTraceError("");
    const traceId = message.metadata?.trace_id;
    if (!traceId && !message.id) return () => { active = false; };
    setTraceLoading(true);
    const request = traceId
      ? getTrace(traceId)
      : getChatMessageTrace(message.id, popup.versionNumber || message.latest_version_number || null);
    request
      .then((report) => { if (active) setTraceReport(report); })
      .catch((error) => { if (active && traceId) setTraceError(error.message); })
      .finally(() => { if (active) setTraceLoading(false); });
    return () => { active = false; };
  }, [message.id, message.metadata?.trace_id, message.latest_version_number, popup.versionNumber, type]);

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
  const selectedSource = filteredContexts.find((context) => context.id === activeSourceId) || filteredContexts[0];
  const traceItems = traceSteps.map((step, index) => ({ step, index }));
  const filteredTraceItems = traceItems.filter(({ step }) => {
    const haystack = [step.step, step.status, step.detail, JSON.stringify(step.metadata || {})].join(" ").toLowerCase();
    return haystack.includes(traceQuery.trim().toLowerCase());
  });
  const selectedTraceItem = filteredTraceItems.find((item) => item.index === selectedTraceIndex) || filteredTraceItems[0];
  const selectedTrace = selectedTraceItem?.step;
  const selectedSpan = selectedTrace?.span;
  const retrievalCandidates = traceRetrievalCandidates(selectedSpan);

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
                    onClick={() => setActiveSourceId(context.id)}
                  >
                    <span>[{context.metadata?.source_label || `S${context.rank}`}] {context.metadata?.source_type || "Chunk"}</span>
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
                        <small>{sourceLabel || selectedSource.metadata?.source_label || `S${selectedSource.rank}`}</small>
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
              <h3><IconLabel icon={GitBranch}>Execution spans</IconLabel></h3>
              <label className="trace-side-search">
                <input value={traceQuery} onChange={(event) => setTraceQuery(event.target.value)} placeholder="Search trace" />
              </label>
              {traceLoading && <p className="muted-text">Loading durable trace...</p>}
              {traceError && <p className="trace-load-warning">{traceError} Showing embedded trace steps.</p>}
              {filteredTraceItems.length === 0 ? (
                <p className="muted-text">No trace steps match.</p>
              ) : filteredTraceItems.map(({ step, index }) => (
                <button key={`${step.step}-${index}`} className={index === selectedTraceItem?.index ? "active" : ""} type="button" onClick={() => setSelectedTraceIndex(index)}>
                  <span>{index + 1}. {step.step}</span>
                  <em>{formatTraceDuration(step.span?.duration_ms || step.metadata?.duration_ms)} · {step.status}</em>
                </button>
              ))}
            </aside>
            <div className="trace-content">
              <div className="trace-toolbar">
                <div className="trace-toolbar-header">
                  <button
                    type="button"
                    className="trace-overview-toggle"
                    aria-expanded={!traceOverviewCollapsed}
                    aria-controls="trace-overview-content"
                    onClick={() => setTraceOverviewCollapsed((collapsed) => !collapsed)}
                  >
                    <IconOnly icon={ChevronRight} />
                    <span>Trace overview</span>
                    <em>{executionSpans.length || traceSteps.length} steps</em>
                  </button>
                  <div className="trace-toolbar-actions">
                    <button type="button" className="secondary-action compact-action" onClick={() => copyText(JSON.stringify(traceReport || traceSteps, null, 2))}><IconLabel icon={Copy}>Copy JSON</IconLabel></button>
                    <button type="button" className="secondary-action compact-action" disabled={!traceReport?.trace_id} onClick={() => downloadTrace(traceReport.trace_id)}><IconLabel icon={Download}>Download JSON</IconLabel></button>
                  </div>
                </div>
                {!traceOverviewCollapsed && (
                  <div className="trace-overview-content" id="trace-overview-content">
                    <TraceSummary metadata={message.metadata || {}} report={traceReport} />
                    {executionSpans.length > 0 && <TraceWaterfall spans={durableSpans} selectedSpanId={selectedSpan?.span_id} onSelect={(spanId) => {
                      const index = executionSpans.findIndex((span) => span.span_id === spanId);
                      if (index >= 0) setSelectedTraceIndex(index);
                    }} />}
                  </div>
                )}
              </div>
              {selectedTrace ? (
                <article className="trace-step-detail">
                  <header>
                    <div>
                      <p className="eyebrow">Selected step</p>
                      <h3>{selectedTrace.step}</h3>
                    </div>
                    <span className="status-pill">{formatTraceDuration(selectedSpan?.duration_ms || selectedTrace.metadata?.duration_ms)} · {selectedTrace.status}</span>
                  </header>
                  <p>{selectedTrace.detail}</p>
                  {selectedSpan ? (
                    <>
                      <div className="trace-detail-tabs" role="tablist">
                        {["inputs", "outputs", "metrics", "model usage", "raw json"].map((tab) => (
                          <button key={tab} type="button" className={traceTab === tab ? "active" : ""} onClick={() => setTraceTab(tab)}>{tab}</button>
                        ))}
                      </div>
                      {traceTab === "inputs" && <TraceJsonPanel value={selectedSpan.input || {}} empty="No input payload recorded." />}
                      {traceTab === "outputs" && (
                        <>
                          {retrievalCandidates.length > 0 && <TraceRetrievalTable candidates={retrievalCandidates} onOpenSource={onOpenSource} />}
                          <TraceJsonPanel value={selectedSpan.output || {}} empty="No output payload recorded." />
                        </>
                      )}
                      {traceTab === "metrics" && <TraceFacts value={{ duration_ms: selectedSpan.duration_ms, ...(selectedSpan.metrics || {}) }} />}
                      {traceTab === "model usage" && <TraceJsonPanel value={{ usage_event_ids: selectedSpan.model_usage_event_ids || [], metrics: selectedSpan.metrics || {} }} empty="No Model Gateway usage event was linked." />}
                      {traceTab === "raw json" && <TraceJsonPanel value={selectedSpan} />}
                    </>
                  ) : (
                    <>
                      <div className="trace-metadata-grid">
                        {Object.entries(selectedTrace.metadata || {}).map(([key, value]) => (
                          <div key={key}><dt>{key}</dt><dd>{formatMetadataValue(value)}</dd></div>
                        ))}
                      </div>
                      <TraceJsonPanel value={selectedTrace} />
                    </>
                  )}
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

function TraceSummary({ metadata, report }) {
  const classifier = metadata.actual_classifier || {};
  const planner = metadata.actual_planner || {};
  const embedding = metadata.query_embedding || {};
  const citationStatus = metadata.citation_validation?.status || (metadata.citations_enabled ? "-" : "disabled");
  const summary = report?.summary || metadata.trace_summary || {};
  return (
    <section className="trace-summary-card">
      <div><dt>Route</dt><dd>{metadata.route_label || metadata.route_level || "-"}</dd></div>
      <div><dt>Predicted complexity</dt><dd>{metadata.predicted_complexity_label || metadata.complexity_label || "-"}</dd></div>
      <div><dt>Routed complexity</dt><dd>{metadata.routed_complexity_label || metadata.complexity_label || "-"}</dd></div>
      <div><dt>Confidence / margin</dt><dd>{formatClassifierScore(metadata.classifier_confidence)} / {formatClassifierScore(metadata.classifier_margin)}</dd></div>
      <div>
        <dt>Classifier</dt>
        <dd>{classifier.name || classifier.model || classifier.runtime || "-"}</dd>
      </div>
      <div>
        <dt>Classifier fallback</dt>
        <dd>{metadata.classifier_fallback_used ? "Yes" : "No"}</dd>
      </div>
      <div>
        <dt>Conversation context</dt>
        <dd>{metadata.history_exchange_count ?? 0} / {metadata.history_exchange_limit ?? "-"} exchange(s)</dd>
      </div>
      <div>
        <dt>History characters</dt>
        <dd>{metadata.history_character_count ?? 0} / {metadata.history_character_limit ?? "-"}</dd>
      </div>
      <div><dt>Query rewritten</dt><dd>{metadata.query_rewritten ? "Yes" : "No"}</dd></div>
      <div><dt>Retrieval</dt><dd>{metadata.retrieval_mode || "none"}</dd></div>
      <div>
        <dt>Query embedding</dt>
        <dd>{embedding.used ? (embedding.model || embedding.deployment_id || "-") : "Not used"}</dd>
      </div>
      <div><dt>Top K</dt><dd>{metadata.top_k ?? "-"}</dd></div>
      <div><dt>Multi-step</dt><dd>{metadata.multi_step ? "Yes" : "No"}</dd></div>
      <div><dt>Subqueries</dt><dd>{metadata.decomposed_queries?.length || 0}</dd></div>
      <div><dt>Agent iterations</dt><dd>{metadata.agent_iterations || 0}</dd></div>
      <div><dt>Agent tool calls</dt><dd>{metadata.agent_tool_calls || 0}</dd></div>
      <div><dt>Agent stop</dt><dd>{metadata.agent_stopping_reason || "-"}</dd></div>
      <div>
        <dt>Planner</dt>
        <dd>{planner.name || planner.model || planner.runtime || "Deterministic"}</dd>
      </div>
      <div><dt>Citations</dt><dd>{citationStatus}</dd></div>
      <div><dt>Latency</dt><dd>{metadata.latency_ms ? `${metadata.latency_ms} ms` : "-"}</dd></div>
      <div><dt>Total trace</dt><dd>{formatTraceDuration(summary.duration_ms)}</dd></div>
      <div><dt>Tokens</dt><dd>{summary.total_tokens ?? "-"}</dd></div>
      <div><dt>Cost</dt><dd>${Number(summary.estimated_cost_usd || 0).toFixed(6)}</dd></div>
      <div><dt>Warnings</dt><dd>{summary.warning_count ?? 0}</dd></div>
      <div><dt>Knowledge base</dt><dd>{metadata.knowledge_base_name || "-"}</dd></div>
    </section>
  );
}

function TraceWaterfall({ spans, selectedSpanId, onSelect }) {
  const durations = spans.map((span) => Number(span.duration_ms || 0));
  const total = Math.max(Number(spans.at(-1)?.finished_at ? new Date(spans.at(-1).finished_at) - new Date(spans[0]?.started_at) : 0), ...durations, 1);
  const origin = new Date(spans[0]?.started_at || 0).getTime();
  const isOrchestrationSpan = (span) => span.category === "orchestration" && !span.parent_span_id;
  const executionSpanCount = spans.filter((span) => !isOrchestrationSpan(span)).length;
  return (
    <section className="trace-waterfall" aria-label="Latency waterfall">
      <header><strong>Latency waterfall</strong><span>{executionSpanCount} steps + total</span></header>
      <div className="trace-waterfall-rows">
        {spans.map((span, index) => {
          const offset = Math.max(new Date(span.started_at).getTime() - origin, 0);
          const barStyle = { marginLeft: `${Math.min((offset / total) * 100, 94)}%`, width: `${Math.max((Number(span.duration_ms || 0) / total) * 100, 1)}%` };
          if (isOrchestrationSpan(span)) {
            return (
              <div key={span.span_id} className="trace-waterfall-total" title={`Total execution: ${formatTraceDuration(span.duration_ms)}`}>
                <span>Total</span>
                <i style={barStyle} />
              </div>
            );
          }
          const stepNumber = spans.slice(0, index + 1).filter((item) => !isOrchestrationSpan(item)).length;
          return (
            <button key={span.span_id} type="button" className={span.span_id === selectedSpanId ? "active" : ""} onClick={() => onSelect(span.span_id)} title={`${span.name}: ${formatTraceDuration(span.duration_ms)}`}>
              <span>{stepNumber}</span>
              <i style={barStyle} />
            </button>
          );
        })}
      </div>
    </section>
  );
}

function TraceJsonPanel({ value, empty = "" }) {
  if (empty && (!value || Object.keys(value).length === 0)) return <p className="muted-text">{empty}</p>;
  return <pre className="trace-json-view">{JSON.stringify(value, null, 2)}</pre>;
}

function TraceFacts({ value }) {
  return <dl className="trace-metadata-grid">{Object.entries(value || {}).map(([key, item]) => <div key={key}><dt>{key}</dt><dd>{formatMetadataValue(item)}</dd></div>)}</dl>;
}

function TraceRetrievalTable({ candidates, onOpenSource }) {
  return (
    <div className="trace-retrieval-table-wrap">
      <table className="trace-retrieval-table">
        <thead><tr><th>Selected</th><th>Chunk</th><th>BM25 raw</th><th>BM25 norm</th><th>Dense</th><th>Dense norm</th><th>Hybrid</th><th>Rank</th></tr></thead>
        <tbody>{candidates.map((candidate, index) => (
          <tr key={candidate.chunk_id || index} className={candidate.selected ? "selected" : ""}>
            <td>{candidate.selected ? "Yes" : "No"}</td><td title={candidate.document_id}><button type="button" onClick={() => onOpenSource?.(candidate)}>{candidate.chunk_id || "-"}</button></td>
            <td>{formatTraceScore(candidate.bm25_raw_score)}</td><td>{formatTraceScore(candidate.bm25_normalized_score)}</td>
            <td>{formatTraceScore(candidate.dense_raw_score)}</td><td>{formatTraceScore(candidate.dense_normalized_score)}</td>
            <td>{formatTraceScore(candidate.hybrid_score)}</td><td>{candidate.selected_rank ?? "-"}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function traceRetrievalCandidates(span) {
  if (!span?.output) return [];
  if (Array.isArray(span.output.candidates)) return span.output.candidates;
  const steps = span.output.diagnostics?.steps;
  if (!Array.isArray(steps)) return [];
  return steps.flatMap((step) => Array.isArray(step.candidates) ? step.candidates.map((candidate) => ({ ...candidate, retrieval_step: step.retrieval_step })) : []);
}

function formatTraceDuration(value) {
  const milliseconds = Number(value || 0);
  if (milliseconds >= 1000) return `${(milliseconds / 1000).toFixed(2)} s`;
  return `${milliseconds.toFixed(milliseconds < 10 ? 2 : 0)} ms`;
}

function formatTraceScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(4) : "-";
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
  const defaultWeights = {
    context_recall: 30,
    factuality: 30,
    token_f1: 10,
    bleu: 5,
    rouge_1: 15,
    rouge_2: 10
  };
  const [knowledgeBases, setKnowledgeBases] = useState([]);
  const [configurations, setConfigurations] = useState([]);
  const [judges, setJudges] = useState([]);
  const [datasets, setDatasets] = useState([]);
  const [experiments, setExperiments] = useState([]);
  const [runs, setRuns] = useState([]);
  const [cases, setCases] = useState([]);
  const [selectedExperimentId, setSelectedExperimentId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [view, setView] = useState("setup");
  const [status, setStatus] = useState("");
  const [isBusy, setIsBusy] = useState(false);
  const [diagnosticLimit, setDiagnosticLimit] = useState(100);
  const [form, setForm] = useState({
    name: "WixQA configuration benchmark",
    knowledgeBaseId: selectedKnowledgeBaseId || "",
    configurationIds: [],
    judgeDeploymentId: "",
    datasetLimits: {},
    weights: defaultWeights,
    maxCostPerCase: "",
    maxAverageLatencyMs: "",
    seed: 42
  });

  const selectedExperiment = experiments.find((item) => item.id === selectedExperimentId);
  const selectedRun = runs.find((item) => item.id === selectedRunId);
  const selectedRagxplain = selectedRun?.metadata?.ragxplain || {};
  const ragxplainJudgeMatches = Boolean(
    selectedRagxplain.status === "completed"
    && selectedRagxplain.judge
    && selectedRagxplain.judge === selectedRun?.metadata?.judge_deployment_id
  );
  const selectedKnowledgeBase = knowledgeBases.find((item) => item.id === form.knowledgeBaseId);
  const activeExperiment = selectedExperiment && ["queued", "running"].includes(selectedExperiment.status);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [nextKbs, nextConfigs, nextJudges, nextDatasets, nextExperiments] = await Promise.all([
          listKnowledgeBases(),
          listChatConfigurations(),
          listModelDeployments({ capability: "judge", enabled: true }).catch(() => []),
          listEvaluationDatasets(),
          listEvaluationExperiments()
        ]);
        if (cancelled) return;
        setKnowledgeBases(nextKbs);
        setConfigurations(nextConfigs);
        setJudges(nextJudges);
        setDatasets(nextDatasets);
        setExperiments(nextExperiments);
        setSelectedExperimentId((current) => current || nextExperiments[0]?.id || "");
        setForm((current) => ({
          ...current,
          knowledgeBaseId: current.knowledgeBaseId || selectedKnowledgeBaseId || nextKbs[0]?.id || "",
          configurationIds: current.configurationIds.length ? current.configurationIds : nextConfigs.slice(0, 1).map((item) => item.id),
          judgeDeploymentId: current.judgeDeploymentId || nextJudges[0]?.id || "",
          datasetLimits: Object.keys(current.datasetLimits).length
            ? current.datasetLimits
            : Object.fromEntries(nextDatasets.map((item) => [item.id, Math.min(item.record_count, item.id === "synthetic" ? 100 : 20)]))
        }));
      } catch (error) {
        if (!cancelled) setStatus(`Evaluation data load failed: ${error.message}`);
      }
    }
    load();
    return () => { cancelled = true; };
  }, [selectedKnowledgeBaseId]);

  useEffect(() => {
    let cancelled = false;
    async function loadExperimentRuns() {
      if (!selectedExperimentId) {
        setRuns([]);
        return;
      }
      try {
        const nextRuns = await listEvaluationExperimentRuns(selectedExperimentId);
        if (cancelled) return;
        setRuns(nextRuns);
        setSelectedRunId((current) => nextRuns.some((run) => run.id === current) ? current : nextRuns[0]?.id || "");
      } catch (error) {
        if (!cancelled) setStatus(`Experiment runs load failed: ${error.message}`);
      }
    }
    loadExperimentRuns();
    return () => { cancelled = true; };
  }, [selectedExperimentId, selectedExperiment?.updated_at]);

  useEffect(() => {
    let cancelled = false;
    async function loadRunCases() {
      if (!selectedRunId) {
        setCases([]);
        return;
      }
      try {
        const nextCases = await listEvaluationCases(selectedRunId);
        if (!cancelled) setCases(nextCases);
      } catch (error) {
        if (!cancelled) setStatus(`Evaluation cases load failed: ${error.message}`);
      }
    }
    loadRunCases();
    return () => { cancelled = true; };
  }, [selectedRunId]);

  useEffect(() => {
    if (!activeExperiment) return undefined;
    const timer = window.setInterval(async () => {
      try {
        const updated = await getEvaluationExperiment(selectedExperiment.id);
        setExperiments((current) => current.map((item) => item.id === updated.id ? updated : item));
      } catch (error) {
        setStatus(`Experiment refresh failed: ${error.message}`);
      }
    }, 2000);
    return () => window.clearInterval(timer);
  }, [activeExperiment, selectedExperiment?.id]);

  function toggleConfiguration(configurationId) {
    setForm((current) => ({
      ...current,
      configurationIds: current.configurationIds.includes(configurationId)
        ? current.configurationIds.filter((id) => id !== configurationId)
        : [...current.configurationIds, configurationId]
    }));
  }

  function setDatasetEnabled(dataset, enabled) {
    setForm((current) => {
      const datasetLimits = { ...current.datasetLimits };
      if (enabled) datasetLimits[dataset.id] = Math.min(dataset.record_count, dataset.id === "synthetic" ? 100 : 20);
      else delete datasetLimits[dataset.id];
      return { ...current, datasetLimits };
    });
  }

  async function startExperiment() {
    if (!form.knowledgeBaseId || !form.configurationIds.length || !form.judgeDeploymentId || !Object.keys(form.datasetLimits).length) {
      setStatus("Select a Knowledge Base, at least one configuration and dataset, and a judge model.");
      return;
    }
    const weightTotal = Object.values(form.weights).reduce((sum, value) => sum + Number(value || 0), 0);
    if (Math.abs(weightTotal - 100) > 0.01) {
      setStatus("Quality metric weights must total 100%.");
      return;
    }
    setIsBusy(true);
    try {
      const created = await createEvaluationExperiment({
        name: form.name,
        knowledge_base_id: form.knowledgeBaseId,
        configuration_ids: form.configurationIds,
        datasets: Object.fromEntries(Object.entries(form.datasetLimits).map(([key, value]) => [key, Number(value) || null])),
        judge_deployment_id: form.judgeDeploymentId,
        quality_weights: Object.fromEntries(Object.entries(form.weights).map(([key, value]) => [key, Number(value) / 100])),
        max_cost_per_case: form.maxCostPerCase === "" ? null : Number(form.maxCostPerCase),
        max_average_latency_ms: form.maxAverageLatencyMs === "" ? null : Number(form.maxAverageLatencyMs),
        seed: Number(form.seed)
      });
      setExperiments((current) => [created.experiment, ...current]);
      setSelectedExperimentId(created.experiment.id);
      setView("leaderboard");
      onSelectKnowledgeBase(form.knowledgeBaseId);
      setStatus("Experiment queued. Keep the Aragbiz worker running to process it.");
    } catch (error) {
      setStatus(`Experiment creation failed: ${error.message}`);
    } finally {
      setIsBusy(false);
    }
  }

  async function cancelSelected() {
    if (!selectedExperiment) return;
    try {
      const updated = await cancelEvaluationExperiment(selectedExperiment.id);
      setExperiments((current) => current.map((item) => item.id === updated.id ? updated : item));
      setStatus("Cancellation requested.");
    } catch (error) {
      setStatus(`Cancellation failed: ${error.message}`);
    }
  }

  async function resumeSelected() {
    if (!selectedExperiment) return;
    try {
      await resumeEvaluationExperiment(selectedExperiment.id);
      setStatus("Experiment resume job queued.");
    } catch (error) {
      setStatus(`Resume failed: ${error.message}`);
    }
  }

  async function removeSelectedExperiment() {
    if (!selectedExperiment) return;
    const confirmed = await confirmAction({
      title: "Delete evaluation experiment?",
      message: `Delete "${selectedExperiment.name}" and all child runs?`,
      detail: "Case results and RAGXplain artifacts for this experiment will be removed.",
      confirmLabel: "Delete experiment"
    });
    if (!confirmed) return;
    try {
      await deleteEvaluationExperiment(selectedExperiment.id);
      const next = experiments.filter((item) => item.id !== selectedExperiment.id);
      setExperiments(next);
      setSelectedExperimentId(next[0]?.id || "");
      setStatus("Evaluation experiment deleted.");
    } catch (error) {
      setStatus(`Delete failed: ${error.message}`);
    }
  }

  async function runRagxplain() {
    if (!selectedRun) return;
    try {
      const queued = await startRagxplainDiagnosis(selectedRun.id, { limit: Number(diagnosticLimit), seed: 42 });
      setRuns((current) => current.map((run) => run.id === selectedRun.id
        ? { ...run, metadata: { ...run.metadata, ragxplain: queued.ragxplain } }
        : run));
      setStatus("RAGXplain diagnosis queued. The worker will create the insights artifacts.");
    } catch (error) {
      setStatus(`RAGXplain diagnosis failed: ${error.message}`);
    }
  }

  async function refreshSelectedRun() {
    if (!selectedRunId) return;
    try {
      const updated = await getEvaluationRun(selectedRunId);
      setRuns((current) => current.map((run) => run.id === updated.id ? updated : run));
      setStatus(`Run refreshed. RAGXplain status: ${updated.metadata?.ragxplain?.status || "not requested"}.`);
    } catch (error) {
      setStatus(`Run refresh failed: ${error.message}`);
    }
  }

  const compatibility = selectedExperiment?.metadata?.knowledge_base_compatibility;
  return (
    <section className="page-stack evaluation-workbench">
      <PanelHeader eyebrow="WixQA Benchmark" title="Evaluation experiments" />
      <div className="tabs evaluation-view-tabs">
        <button className={view === "setup" ? "active" : ""} onClick={() => setView("setup")}>Setup</button>
        <button className={view === "leaderboard" ? "active" : ""} onClick={() => setView("leaderboard")}>Progress & leaderboard</button>
        <button className={view === "diagnostics" ? "active" : ""} onClick={() => setView("diagnostics")}>Diagnostics</button>
      </div>
      {status && <div className="inline-status">{status}</div>}

      {view === "setup" && (
        <div className="evaluation-setup-grid">
          <section className="panel evaluation-setup-panel">
            <h2>Experiment</h2>
            <label>Experiment name<input value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} /></label>
            <SelectField
              label="WixQA Knowledge Base"
              value={form.knowledgeBaseId}
              options={[{ value: "", label: "Select Knowledge Base" }, ...knowledgeBases.map((item) => ({ value: item.id, label: item.name }))]}
              onChange={(knowledgeBaseId) => setForm({ ...form, knowledgeBaseId })}
            />
            {selectedKnowledgeBase && (
              <div className={`evaluation-compatibility ${selectedKnowledgeBase.document_count === 6221 ? "is-compatible" : "is-warning"}`}>
                <strong>{selectedKnowledgeBase.document_count === 6221 ? "WixQA corpus size matched" : "Knowledge Base compatibility warning"}</strong>
                <span>{selectedKnowledgeBase.document_count} documents, {selectedKnowledgeBase.chunk_count} chunks, {selectedKnowledgeBase.embedding_model}</span>
              </div>
            )}
            <SelectField
              label="Shared LLM judge"
              value={form.judgeDeploymentId}
              options={[{ value: "", label: "Select enabled judge model" }, ...judges.map((item) => deploymentOption(item))]}
              onChange={(judgeDeploymentId) => setForm({ ...form, judgeDeploymentId })}
            />
            <label>Sampling seed<input type="number" value={form.seed} onChange={(event) => setForm({ ...form, seed: Number(event.target.value) })} /></label>
          </section>
          <section className="panel evaluation-setup-panel">
            <div className="panel-heading-row"><h2>RAG Customizer configurations</h2><button className="text-action" type="button" onClick={() => setForm({ ...form, configurationIds: configurations.map((item) => item.id) })}>Select all</button></div>
            <div className="evaluation-check-list">
              {configurations.map((configuration) => (
                <label className="check-row" key={configuration.id}>
                  <input type="checkbox" checked={form.configurationIds.includes(configuration.id)} onChange={() => toggleConfiguration(configuration.id)} />
                  <span>
                    <strong>{configuration.metadata?.configuration_id || configuration.id} | {configuration.name}</strong>
                    <small>{configuration.metadata?.route_strategy || configuration.metadata?.route_mode || "Adaptive"} | {configuration.generator_provider} / {configuration.generator_model}</small>
                  </span>
                </label>
              ))}
              {!configurations.length && <p className="muted-text">Create saved RAG Customizer configurations before benchmarking.</p>}
            </div>
          </section>
          <section className="panel evaluation-setup-panel">
            <h2>WixQA datasets</h2>
            <div className="evaluation-dataset-list">
              {datasets.map((dataset) => {
                const enabled = Object.prototype.hasOwnProperty.call(form.datasetLimits, dataset.id);
                return (
                  <div className="evaluation-dataset-row" key={dataset.id}>
                    <label className="check-row">
                      <input type="checkbox" checked={enabled} onChange={(event) => setDatasetEnabled(dataset, event.target.checked)} />
                      <span><strong>{dataset.name}</strong><small>{dataset.record_count.toLocaleString()} records</small></span>
                    </label>
                    <label>Limit<input disabled={!enabled} type="number" min="1" max={dataset.record_count} value={form.datasetLimits[dataset.id] ?? ""} onChange={(event) => setForm({ ...form, datasetLimits: { ...form.datasetLimits, [dataset.id]: Number(event.target.value) } })} /></label>
                  </div>
                );
              })}
            </div>
          </section>
          <section className="panel evaluation-setup-panel">
            <h2>Quality score & constraints</h2>
            <div className="evaluation-weight-grid">
              {Object.entries(form.weights).map(([metric, value]) => (
                <label key={metric}>{metric.replaceAll("_", " ")}<input type="number" min="0" max="100" value={value} onChange={(event) => setForm({ ...form, weights: { ...form.weights, [metric]: Number(event.target.value) } })} /></label>
              ))}
            </div>
            <p className="muted-text">Weight total: {Object.values(form.weights).reduce((sum, value) => sum + Number(value || 0), 0)}%</p>
            <div className="config-two-column">
              <label>Max cost / case (USD)<input type="number" min="0" step="0.001" placeholder="No limit" value={form.maxCostPerCase} onChange={(event) => setForm({ ...form, maxCostPerCase: event.target.value })} /></label>
              <label>Max average latency (ms)<input type="number" min="0" placeholder="No limit" value={form.maxAverageLatencyMs} onChange={(event) => setForm({ ...form, maxAverageLatencyMs: event.target.value })} /></label>
            </div>
            <button className="primary-action" type="button" onClick={startExperiment} disabled={isBusy}>
              <IconLabel icon={ClipboardList}>{isBusy ? "Creating..." : "Start benchmark"}</IconLabel>
            </button>
          </section>
        </div>
      )}

      {view === "leaderboard" && (
        <div className="evaluation-results-layout">
          <aside className="panel evaluation-experiment-list">
            <div className="panel-heading-row"><h2>Experiments</h2><button className="icon-button" aria-label="Refresh experiments" onClick={async () => setExperiments(await listEvaluationExperiments())}><RefreshCw size={16} /></button></div>
            {experiments.map((experiment) => (
              <button key={experiment.id} className={`evaluation-experiment-item ${experiment.id === selectedExperimentId ? "selected" : ""}`} onClick={() => setSelectedExperimentId(experiment.id)}>
                <strong>{experiment.name}</strong>
                <span>{experiment.status} - {experiment.progress?.percent || 0}%</span>
              </button>
            ))}
          </aside>
          <section className="panel evaluation-leaderboard-panel">
            {!selectedExperiment ? (
              <div className="empty-state"><strong>No experiment selected</strong><p>Create a WixQA benchmark experiment first.</p></div>
            ) : (
              <>
                <div className="panel-heading-row">
                  <div><h2>{selectedExperiment.name}</h2><p>{selectedExperiment.knowledge_base_name}</p></div>
                  <div className="action-row">
                    {activeExperiment && <button className="secondary-action" onClick={cancelSelected}>Cancel</button>}
                    {["partial", "failed", "cancelled"].includes(selectedExperiment.status) && <button className="secondary-action" onClick={resumeSelected}>Resume</button>}
                    <button className="secondary-action danger-action" onClick={removeSelectedExperiment}><Trash2 size={16} /> Delete</button>
                  </div>
                </div>
                <div className="evaluation-progress-track"><span style={{ width: `${selectedExperiment.progress?.percent || 0}%` }} /></div>
                <p className="muted-text">{selectedExperiment.progress?.completed_cells || 0} / {selectedExperiment.progress?.total_cells || 0} configuration-dataset cells</p>
                {compatibility?.status === "warning" && <div className="evaluation-compatibility is-warning"><strong>Compatibility warning</strong><span>{compatibility.message}</span></div>}
                <div className="table-panel">
                  <table>
                    <thead><tr><th>Rank</th><th>Configuration</th><th>Quality</th><th>Cost / case</th><th>Latency</th><th>Eligibility</th></tr></thead>
                    <tbody>
                      {(selectedExperiment.leaderboard || []).map((entry) => (
                        <tr key={entry.configuration_id} className={entry.winner ? "evaluation-winner" : ""}>
                          <td>{entry.winner ? "Best" : entry.rank || "-"}</td>
                          <td><strong>{entry.configuration_name}</strong><small>{entry.configuration_route} | {Object.entries(entry.dataset_scores || {}).map(([name, score]) => `${name}: ${formatPercentMetric(score)}`).join(" | ")}</small></td>
                          <td>{formatPercentMetric(entry.quality_score)}</td>
                          <td>${formatNumber(entry.average_cost_per_case_usd, 4)}</td>
                          <td>{formatNumber(entry.average_latency_ms)} ms</td>
                          <td>{entry.eligible ? "Eligible" : `Excluded: ${(entry.constraint_violations || []).join(", ")}`}</td>
                        </tr>
                      ))}
                      {!selectedExperiment.leaderboard?.length && <tr><td colSpan="6">The leaderboard will appear when the worker completes this experiment.</td></tr>}
                    </tbody>
                  </table>
                </div>
                <div className="evaluation-cell-grid">
                  {runs.map((run) => (
                    <button key={run.id} className={`evaluation-cell ${run.id === selectedRunId ? "selected" : ""}`} onClick={() => { setSelectedRunId(run.id); setView("diagnostics"); }}>
                      <strong>{run.name}</strong>
                      <span>{formatPercentMetric(run.metrics?.wixqa?.context_recall)} recall - {formatPercentMetric(run.metrics?.wixqa?.factuality)} factuality</span>
                    </button>
                  ))}
                </div>
              </>
            )}
          </section>
        </div>
      )}

      {view === "diagnostics" && (
        <section className="panel evaluation-diagnostics-panel">
          <div className="panel-heading-row">
            <div>
              <h2>{selectedRun?.name || "Select a configuration-dataset result"}</h2>
              <p>{selectedRun ? `${selectedRun.metadata?.record_count || 0} cases` : "Choose a result from Progress & leaderboard."}</p>
            </div>
            {selectedRun && (
              <div className="action-row">
                <label className="compact-number-field">RAGXplain cases<input type="number" min="1" max={selectedRun.metadata?.record_count || 100} value={diagnosticLimit} onChange={(event) => setDiagnosticLimit(Number(event.target.value))} /></label>
                <button className="secondary-action" onClick={runRagxplain}><GitBranch size={16} /> Run diagnosis</button>
                <button className="icon-button" aria-label="Refresh selected evaluation run" title="Refresh RAGXplain status" onClick={refreshSelectedRun}><RefreshCw size={16} /></button>
                <button className="secondary-action" disabled={!ragxplainJudgeMatches} onClick={() => onOpenDetail(selectedRun, null, "ragxplain")}><ExternalLink size={16} /> Open insights</button>
              </div>
            )}
          </div>
          {selectedRun?.metadata?.ragxplain && (
            <div className={`evaluation-ragxplain-status ${!ragxplainJudgeMatches && selectedRagxplain.status === "completed" ? "is-failed" : `is-${selectedRagxplain.status || "not-requested"}`}`}>
              <strong>RAGXplain: {selectedRagxplain.status || "not requested"}</strong>
              {!ragxplainJudgeMatches && selectedRagxplain.status === "completed" && (
                <span>Legacy artifact used {selectedRagxplain.judge || "an unknown judge"}. Run diagnosis again to use {selectedRun.metadata?.judge_deployment_id}.</span>
              )}
              {selectedRagxplain.error && <span>{selectedRagxplain.error}</span>}
            </div>
          )}
          {selectedRun && (
            <div className="metrics-grid evaluation-metrics-grid">
              <article className="metric-card"><small>Token F1</small><strong>{formatPercentMetric(selectedRun.metrics?.wixqa?.token_f1)}</strong><span>WixQA</span></article>
              <article className="metric-card"><small>BLEU</small><strong>{formatPercentMetric(selectedRun.metrics?.wixqa?.bleu)}</strong><span>WixQA</span></article>
              <article className="metric-card"><small>ROUGE-1 / 2</small><strong>{formatPercentMetric(selectedRun.metrics?.wixqa?.rouge_1)} / {formatPercentMetric(selectedRun.metrics?.wixqa?.rouge_2)}</strong><span>WixQA</span></article>
              <article className="metric-card"><small>Context Recall</small><strong>{formatPercentMetric(selectedRun.metrics?.wixqa?.context_recall)}</strong><span>LLM judge</span></article>
              <article className="metric-card"><small>Factuality</small><strong>{formatPercentMetric(selectedRun.metrics?.wixqa?.factuality)}</strong><span>LLM judge</span></article>
            </div>
          )}
          <div className="table-panel evaluation-case-table">
            <table>
              <thead><tr><th>Question</th><th>Route</th><th>F1</th><th>Recall</th><th>Factuality</th><th>Detail</th></tr></thead>
              <tbody>
                {cases.map((evaluationCase) => {
                  const wixqa = evaluationCase.metrics?.result?.wixqa || {};
                  return (
                    <tr key={evaluationCase.id}>
                      <td>{evaluationCase.question}</td>
                      <td>{evaluationCase.answer_metadata?.route_label || evaluationCase.answer_metadata?.route_level || "-"}</td>
                      <td>{formatPercentMetric(wixqa.token_f1)}</td>
                      <td>{formatPercentMetric(wixqa.context_recall)}</td>
                      <td>{formatPercentMetric(wixqa.factuality)}</td>
                      <td><button className="secondary-action compact-action" onClick={() => onOpenDetail(selectedRun, evaluationCase, "case")}><GitBranch size={15} /> Trace</button></td>
                    </tr>
                  );
                })}
                {!cases.length && <tr><td colSpan="6">Select a completed configuration-dataset result.</td></tr>}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </section>
  );
}

function EvaluationDetailScreen({ detail, onBack }) {
  const evaluationCase = detail?.evaluationCase;
  const run = detail?.run;
  if (detail?.view === "ragxplain") {
    return <RagxplainInsightsScreen run={run} onBack={onBack} />;
  }
  const traceSteps = Array.isArray(evaluationCase?.answer_metadata?.trace_steps) ? evaluationCase.answer_metadata.trace_steps : [];
  if (!evaluationCase) {
    return (
      <section className="page-stack">
        <PanelHeader eyebrow="Evaluation Detail" title="WixQA case" />
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
      <PanelHeader eyebrow="Evaluation Detail" title="WixQA case" />
      <section className="panel evaluation-case-summary">
        <p className="eyebrow">{run?.name || "Evaluation run"}</p>
        <h2>{evaluationCase.question}</h2>
        <div className="metrics-grid evaluation-metrics-grid">
          <article className="metric-card"><small>Expected label</small><strong>{evaluationCase.complexity_label}</strong><span>benchmark</span></article>
          <article className="metric-card"><small>Executed route</small><strong>{evaluationCase.answer_metadata?.route_label || "-"}</strong><span>{evaluationCase.answer_metadata?.complexity_label || "configured route"}</span></article>
          <article className="metric-card"><small>Context Recall</small><strong>{formatPercentMetric(evaluationCase.metrics?.result?.wixqa?.context_recall)}</strong><span>WixQA judge</span></article>
          <article className="metric-card"><small>Factuality</small><strong>{formatPercentMetric(evaluationCase.metrics?.result?.wixqa?.factuality)}</strong><span>WixQA judge</span></article>
        </div>
      </section>
      <section className="evaluation-answer-grid">
        <article className="panel">
          <h3>Configuration answer</h3>
          <p>{evaluationCase.answer}</p>
        </article>
        <article className="panel">
          <h3>Expected answer</h3>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{evaluationCase.expected_answer}</ReactMarkdown>
        </article>
      </section>
      <section className="panel">
        <PanelHeader eyebrow="Sources" title="Retrieved contexts" />
        <div className="run-list">
          {evaluationCase.contexts.map((context) => (
            <article className="run-card" key={context.id}>
              <div>
                <strong>{context.metadata?.title || context.id}</strong>
                <small>{context.mode} - rank {context.rank} - chunk {context.metadata?.chunk_index ?? "-"}</small>
              </div>
              <p>{context.text}</p>
            </article>
          ))}
          {evaluationCase.contexts.length === 0 && <p className="muted-text">No contexts returned by this configuration.</p>}
        </div>
      </section>
      <section className="panel">
        <PanelHeader eyebrow="Trace" title="Configuration pipeline trace" />
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
      {ragxplain.status !== "completed" && (
        <div className={`evaluation-ragxplain-summary is-${ragxplain.status || "not_requested"}`}>
          <div><strong>RAGXplain {String(ragxplain.status || "not_requested").replace("_", " ")}</strong><span>{ragxplain.judge || "Judge unavailable"}</span></div>
          {ragxplain.error && <p>{ragxplain.error}</p>}
        </div>
      )}
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

function AssistantMessageContent({
  content,
  citationSources = {},
  onCitationClick = () => {},
  onInvalidCitation = () => {}
}) {
  const validLabels = Object.keys(citationSources);
  return (
    <div className="message-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkSourceCitations, { validLabels }]]}
        skipHtml
        components={{
          a: ({ children, href, title }) => {
            if (href?.startsWith("aragbiz-source:")) {
              const label = decodeURIComponent(href.slice("aragbiz-source:".length));
              const source = citationSources[label];
              return (
                <button
                  className="inline-citation"
                  type="button"
                  title={source?.title ? `${label}: ${source.title}` : `Open source ${label}`}
                  onClick={() => onCitationClick(label, source)}
                >
                  [{label}]
                </button>
              );
            }
            if (href?.startsWith("aragbiz-invalid:")) {
              const label = decodeURIComponent(href.slice("aragbiz-invalid:".length));
              return (
                <button
                  className="inline-citation invalid"
                  type="button"
                  title={`${label} was not found in the retrieved sources`}
                  onClick={onInvalidCitation}
                >
                  [{label}]
                </button>
              );
            }
            return (
              <a href={href} title={title} target="_blank" rel="noreferrer">
                {children}
              </a>
            );
          }
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}

function remarkSourceCitations({ validLabels = [] } = {}) {
  const valid = new Set(validLabels);
  return (tree) => {
    transformCitationTextNodes(tree, valid);
  };
}

function transformCitationTextNodes(node, validLabels) {
  if (!node || !Array.isArray(node.children)) return;
  if (["code", "inlineCode", "link", "linkReference"].includes(node.type)) return;
  const nextChildren = [];
  node.children.forEach((child) => {
    if (child.type !== "text") {
      transformCitationTextNodes(child, validLabels);
      nextChildren.push(child);
      return;
    }
    const value = String(child.value || "");
    const pattern = /\[(S\d+)\]/g;
    let cursor = 0;
    let match = pattern.exec(value);
    if (!match) {
      nextChildren.push(child);
      return;
    }
    while (match) {
      if (match.index > cursor) {
        nextChildren.push({ type: "text", value: value.slice(cursor, match.index) });
      }
      const label = match[1];
      nextChildren.push({
        type: "link",
        url: `${validLabels.has(label) ? "aragbiz-source:" : "aragbiz-invalid:"}${encodeURIComponent(label)}`,
        children: [{ type: "text", value: `[${label}]` }]
      });
      cursor = match.index + match[0].length;
      match = pattern.exec(value);
    }
    if (cursor < value.length) {
      nextChildren.push({ type: "text", value: value.slice(cursor) });
    }
  });
  node.children = nextChildren;
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
  return entries.map(([key, value]) => `${key.replace("l1_", "L1 ").replace("l2_", "L2 ").replace("l3_", "L3 ").replace("l4_", "L4 ")}: ${value}`).join(" / ");
}

function answerModeFromRoute(route) {
  if (route === "L1 Direct") return "direct";
  if (route === "L2 Simple RAG") return "simple_rag";
  if (route === "L3 Complex RAG") return "complex_rag";
  if (route === "L4 Advanced RAG") return "advanced_rag";
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
  if (mode === "advanced_rag") return "L4 Advanced RAG";
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
      metadata: {
        ...(record.metadata || {}),
        assistant_message_id: record.id
      },
      status: record.status || "completed",
      streaming: false,
      versionCount: Number(record.version_count || 1),
      latestVersionNumber: Number(record.latest_version_number || record.metadata?.message_version_number || 1),
      latestVersionStatus: record.latest_version_status || record.status || "completed",
      viewingVersionNumber: Number(record.metadata?.message_version_number || 1)
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
    classifierConfidenceThreshold: clampNumber(metadata.classifier_confidence_threshold, 0.6, 0, 1),
    classifierMarginThreshold: clampNumber(metadata.classifier_margin_threshold, 0.15, 0, 1),
    retrievalMode,
    topK: clampNumber(metadata.top_k, current.topK || 6, 1, 50),
    reranker: typeof metadata.reranker_enabled === "boolean" ? metadata.reranker_enabled : current.reranker,
    generatorDeploymentId: snapshot.generator_deployment_id || metadata.generator_deployment_id || defaultChatConfigurationDraft.generatorDeploymentId,
    fallbackDeploymentIds: Array.isArray(snapshot.fallback_deployment_ids) ? snapshot.fallback_deployment_ids : [],
    rerankerDeploymentId: snapshot.reranker_deployment_id || "",
    plannerDeploymentId: snapshot.planner_deployment_id || "",
    agentMaxIterations: clampNumber(metadata.agent_max_iterations, 5, 1, 8),
    agentMaxToolCalls: clampNumber(metadata.agent_max_tool_calls, 8, 1, 12),
    agentTimeoutSeconds: clampNumber(metadata.agent_timeout_seconds, 90, 30, 180),
    agentPublicWebEnabled: Boolean(metadata.agent_public_web_enabled),
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
    conversationAwarenessEnabled: typeof metadata.conversation_awareness_enabled === "boolean"
      ? metadata.conversation_awareness_enabled
      : true,
    conversationHistoryExchanges: clampNumber(
      metadata.conversation_history_exchanges,
      defaultConversationLimits.defaultCompletedExchanges,
      1,
      defaultConversationLimits.maxCompletedExchanges
    ),
    conversationHistoryCharacters: clampNumber(
      metadata.conversation_history_characters,
      defaultConversationLimits.defaultCharacters,
      1,
      defaultConversationLimits.maxCharacters
    ),
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

function chatConfigurationPayloadFromDraft(config, conversationLimits = defaultConversationLimits) {
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
      classifier_confidence_threshold: clampNumber(config.classifierConfidenceThreshold, 0.6, 0, 1),
      classifier_margin_threshold: clampNumber(config.classifierMarginThreshold, 0.15, 0, 1),
      planner_deployment_id: config.plannerDeploymentId || "",
      agent_max_iterations: clampNumber(config.agentMaxIterations, 5, 1, 8),
      agent_max_tool_calls: clampNumber(config.agentMaxToolCalls, 8, 1, 12),
      agent_timeout_seconds: clampNumber(config.agentTimeoutSeconds, 90, 30, 180),
      agent_public_web_enabled: Boolean(config.agentPublicWebEnabled),
      retrieval_mode: retrievalModeValue(config.retrievalMode),
      retrieval_mode_label: config.retrievalMode || "Hybrid",
      top_k: clampNumber(config.topK, 6, 1, 50),
      reranker_enabled: Boolean(config.reranker),
      configuration_id: normalizeConfigurationCode(config.configurationCode) || createConfigurationCode(),
      welcome_message: config.welcomeMessage || defaultChatConfigurationDraft.welcomeMessage,
      conversation_starters: normalizeConversationStarters(config.conversationStarters),
      conversation_awareness_enabled: config.conversationAwarenessEnabled !== false,
      conversation_history_exchanges: clampNumber(
        config.conversationHistoryExchanges,
        conversationLimits.defaultCompletedExchanges,
        1,
        conversationLimits.maxCompletedExchanges
      ),
      conversation_history_characters: clampNumber(
        config.conversationHistoryCharacters,
        conversationLimits.defaultCharacters,
        1,
        conversationLimits.maxCharacters
      ),
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

function formatClassifierScore(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3) : "-";
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

function classifierDeploymentOption(deployment = {}) {
  const option = deploymentOption(deployment);
  const labels = deployment.metadata?.complexity_labels || deployment.metadata_json?.complexity_labels || [];
  const isKnownFourClass = Array.isArray(labels) && labels.includes("advanced");
  const isLegacyClassifier = (
    (Array.isArray(labels) && labels.length === 3)
    || ["query_classifier_distilbert", "query_classifier_t5"].includes(deployment.model)
  );
  const classLabel = isLegacyClassifier ? "legacy 3-class" : (isKnownFourClass ? "4-class" : "4-class contract");
  return {
    ...option,
    label: `${option.label} (${classLabel})`,
    title: `${option.title}. ${isLegacyClassifier ? "Cannot predict advanced." : "Supports simple, moderate, complex, and advanced."}`
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

function normalizeConversationLimits(payload = {}) {
  const maxCompletedExchanges = clampNumber(
    payload.max_completed_exchanges,
    defaultConversationLimits.maxCompletedExchanges,
    1,
    Number.MAX_SAFE_INTEGER
  );
  const maxCharacters = clampNumber(
    payload.max_characters,
    defaultConversationLimits.maxCharacters,
    1,
    Number.MAX_SAFE_INTEGER
  );
  return {
    maxCompletedExchanges,
    maxCharacters,
    defaultCompletedExchanges: clampNumber(
      payload.default_completed_exchanges,
      defaultConversationLimits.defaultCompletedExchanges,
      1,
      maxCompletedExchanges
    ),
    defaultCharacters: clampNumber(
      payload.default_characters,
      defaultConversationLimits.defaultCharacters,
      1,
      maxCharacters
    )
  };
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
