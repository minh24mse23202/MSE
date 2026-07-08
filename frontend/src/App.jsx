import { useEffect, useRef, useState } from "react";
import {
  Activity,
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
  Database,
  FileText,
  Filter,
  GitBranch,
  Globe,
  HardDrive,
  Home,
  Layers,
  LogIn,
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
  createKnowledgeBase,
  deleteChatConversation,
  deleteKnowledgeBase,
  deleteKnowledgeDocument,
  getKnowledgeProcessingTrace,
  ingestWebsiteSource,
  listChatConversations,
  listChatMessages,
  listKnowledgeChunks,
  listKnowledgeDocuments,
  listKnowledgeBases,
  reindexKnowledgeBase,
  submitFeedback,
  updateChatConversation,
  updateKnowledgeBase,
  uploadKnowledgeSource
} from "./api.js";
import {
  architectureLayers,
  evaluationRuns,
  feedbackRows,
  seedMessages,
  tokenStats
} from "./data.js";

const navItems = [
  { id: "main", label: "Main", icon: Home },
  { id: "knowledge", label: "Knowledge Bases", icon: Database },
  { id: "evaluation", label: "Evaluation", icon: ClipboardList },
  { id: "analytics", label: "Analytics", icon: BarChart3 }
];

const classifiers = ["DistilBERT", "T5-small", "Naive Bayes", "Heuristic"];
const routes = [
  { value: "Adaptive", label: "Adaptive" },
  { value: "L1 Direct", label: "L1 Direct Generation" },
  { value: "L2 Simple RAG", label: "L2 Simple RAG" },
  { value: "L3 Complex RAG", label: "L3 Complex RAG" }
];
const generatorProviderOptions = [
  { value: "Local", label: "Local" },
  { value: "OpenAI", label: "OpenAI (coming soon)", disabled: true },
  { value: "Hugging Face", label: "Hugging Face hosted (coming soon)", disabled: true },
  { value: "Cohere", label: "Cohere (coming soon)", disabled: true },
  { value: "AWS Bedrock", label: "AWS Bedrock (coming soon)", disabled: true }
];
const generatorProviders = generatorProviderOptions.map((option) => option.value);
const generatorModelsByProvider = {
  Local: ["extractive", "google/flan-t5-small"],
  OpenAI: ["gpt-4.1-mini", "gpt-4.1", "o4-mini"],
  "Hugging Face": ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.2-3B-Instruct"],
  Cohere: ["command-r", "command-r-plus"],
  "AWS Bedrock": ["anthropic.claude-3-haiku", "amazon.nova-lite", "meta.llama3-8b-instruct"]
};
const generatorModelOptionsByProvider = {
  Local: [
    { value: "extractive", label: "extractive (supported)" },
    { value: "google/flan-t5-small", label: "google/flan-t5-small (local optional)" }
  ],
  OpenAI: generatorModelsByProvider.OpenAI.map((model) => ({ value: model, label: `${model} (coming soon)`, disabled: true })),
  "Hugging Face": generatorModelsByProvider["Hugging Face"].map((model) => ({ value: model, label: `${model} (coming soon)`, disabled: true })),
  Cohere: generatorModelsByProvider.Cohere.map((model) => ({ value: model, label: `${model} (coming soon)`, disabled: true })),
  "AWS Bedrock": generatorModelsByProvider["AWS Bedrock"].map((model) => ({ value: model, label: `${model} (coming soon)`, disabled: true }))
};
const responseStructures = [
  "Concise answer with bullets and cited workflow context",
  "Step-by-step workflow guidance",
  "Executive summary then details",
  "Detailed answer with assumptions and risks",
  "JSON-style structured response"
];
const chatbotTones = ["Professional", "Friendly", "Formal", "Technical", "Coaching"];
const defaultChatConfigurationDraft = {
  chatConfigurationId: "",
  configurationName: "Balanced workflow assistant",
  configurationDescription: "Default configuration for concise business workflow answers.",
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
        <button className="primary-action" onClick={onSubmit}><IconLabel icon={LogIn} size={20}>Login</IconLabel></button>
        <button className="text-action" onClick={onSwitch}>
          <IconLabel icon={UserPlus}>Need an account? Sign up</IconLabel>
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
            <span><IconOnly icon={RefreshCw} size={16} /></span>
            <small>reCAPTCHA</small>
          </div>
        </div>
        <label className="signup-consent">
          <input type="checkbox" />
          <span>
            I agree to use Adaptive RAG Studio as an AI窶叢owered system. I will verify answers since AI can make mistakes.
          </span>
        </label>
        <button className="primary-action signup-submit" onClick={onSubmit}><IconLabel icon={UserPlus} size={20}>Agree and start</IconLabel></button>
        <button className="text-action signup-login-link" onClick={onSwitch}>
          <IconLabel icon={LogIn}>Already have an account? Log in</IconLabel>
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
          <span className="user-chip"><IconLabel icon={CircleUserRound}>Researcher</IconLabel></span>
        </div>
      </header>
      <main className="workspace">{children}</main>
    </div>
  );
}

function MainScreen({ selectedKnowledgeBaseId, onSelectKnowledgeBase }) {
  const mainGridRef = useRef(null);
  const [messages, setMessages] = useState(seedMessages);
  const [question, setQuestion] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [popup, setPopup] = useState(null);
  const [knowledgeBaseOptions, setKnowledgeBaseOptions] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState("");
  const [historyQuery, setHistoryQuery] = useState("");
  const [recentConversations, setRecentConversations] = useState([]);
  const [libraryConversations, setLibraryConversations] = useState([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [chatConfigurations, setChatConfigurations] = useState([]);
  const [configurationStatus, setConfigurationStatus] = useState("");
  const [layout, setLayout] = useState(loadMainLayout);
  const [config, setConfig] = useState({
    classifier: "DistilBERT",
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
  }, []);

  useEffect(() => {
    saveMainLayout(layout);
  }, [layout]);

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
      const selected = items.find((item) => item.id === preferredId) || items[0];
      if (selected && !preferredId) {
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
      setConfig((current) => ({ ...current, chatConfigurationId: "" }));
      return;
    }
    setConfig((current) => applyChatConfigurationToDraft(current, selected));
    setConfigurationStatus(`Loaded "${selected.name}"`);
  }

  async function saveChatConfigurationAsNew() {
    try {
      const created = await createChatConfiguration(chatConfigurationPayloadFromDraft(config));
      const items = await refreshChatConfigurations(created.id);
      setChatConfigurations(items);
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
      const items = await refreshChatConfigurations(updated.id);
      setChatConfigurations(items);
      setConfig((current) => applyChatConfigurationToDraft(current, updated));
      setConfigurationStatus(`Updated "${updated.name}"`);
    } catch (error) {
      setConfigurationStatus(`Update configuration failed: ${error.message}`);
    }
  }

  function resetChatConfigurationDraft() {
    const selected = chatConfigurations.find((item) => item.id === config.chatConfigurationId) || chatConfigurations[0];
    setConfig((current) => selected ? applyChatConfigurationToDraft(current, selected) : { ...current, ...defaultChatConfigurationDraft });
    setConfigurationStatus("Configuration draft reset");
  }

  async function startNewChat() {
    setActiveConversationId("");
    setMessages([]);
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

  async function removeConversation(conversation) {
    const confirmed = window.confirm(`Delete "${conversation.title}"?`);
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
    const userMessage = { id: createId(), role: "user", content: trimmed };
    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setIsLoading(true);
    try {
      const response = await askQuestion(trimmed, {
        conversationId: activeConversationId,
        knowledgeBaseId: selectedKnowledgeBaseId,
        mode,
        retrievalMode: retrievalModeValue(config.retrievalMode),
        topK: config.topK,
        chatConfigurationId: config.chatConfigurationId || null,
        chatConfiguration: chatConfigurationPayloadFromDraft(config)
      });
      if (response.conversation_id && response.conversation_id !== activeConversationId) {
        setActiveConversationId(response.conversation_id);
      }
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
      await refreshConversationLists();
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: createId(),
          question: trimmed,
          role: "assistant",
          content: `Answer request failed: ${error.message}`,
          contexts: [],
          metadata: { error: error.message, complexity_label: "unknown", trace_steps: [] }
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
  const lastAnswer = [...messages].reverse().find((message) => message.role === "assistant" && message.metadata);
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
        selectedRoute={config.route}
        requiresKnowledgeBase={answerModeFromRoute(config.route) !== "direct"}
      />
      <PanelResizeHandle side="right" label="Resize configuration" onPointerDown={(event) => beginPanelResize("config", event)} />
      <RagConfiguration
        config={config}
        setConfig={setConfig}
        selectedKnowledgeBase={selectedKnowledgeBase}
        lastAnswerMetadata={lastAnswer?.metadata}
        chatConfigurations={chatConfigurations}
        configurationStatus={configurationStatus}
        onSelectChatConfiguration={selectChatConfiguration}
        onSaveConfiguration={saveChatConfigurationAsNew}
        onUpdateConfiguration={updateSelectedChatConfiguration}
        onResetConfiguration={resetChatConfigurationDraft}
        collapsed={layout.configCollapsed}
        onToggle={() => togglePanel("config")}
      />
      {feedbackStatus && <div className="toast">{feedbackStatus}</div>}
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
          <p className="eyebrow">Notebook</p>
          <h2>Chat history</h2>
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
  onDeleteConversation
}) {
  return (
    <section className="conversation-section">
      <h3><IconLabel icon={icon}>{title}</IconLabel></h3>
      <div className="conversation-list">
        {conversations.length === 0 ? (
          <p className="history-empty">{emptyText}</p>
        ) : conversations.map((conversation) => (
          <article
            key={conversation.id}
            className={`conversation-item ${conversation.id === activeConversationId ? "active" : ""}`}
          >
            <button className="conversation-main" type="button" onClick={() => onSelectConversation(conversation)}>
              <span>
                <strong>{conversation.title}</strong>
                <small>{conversation.route_mode || "adaptive"} - {conversation.retrieval_mode || "hybrid"}</small>
              </span>
              <em>{formatShortDate(conversation.updated_at)}</em>
            </button>
            <div className="conversation-actions">
              <button
                type="button"
                aria-label={conversation.pinned ? "Unpin chat" : "Pin chat"}
                onClick={() => onTogglePinned(conversation)}
              >
                <IconOnly icon={conversation.pinned ? PinOff : Pin} size={14} />
              </button>
              <button type="button" aria-label="Delete chat" onClick={() => onDeleteConversation(conversation)}>
                <IconOnly icon={Trash2} size={14} />
              </button>
            </div>
          </article>
        ))}
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
  selectedRoute,
  requiresKnowledgeBase
}) {
  return (
    <section className="chat-panel">
      <header className="chat-titlebar">
        <div className="brain-mark"><BrainCircuit size={18} aria-hidden="true" /></div>
        <div>
          <h1>Business Workflow Question Answering</h1>
          <p>Business Workflow Question Answering AI Chatbot</p>
        </div>
      </header>
      <div className="message-list">
        {messages.map((message) => (
          <article key={message.id} className={`message ${message.role}`}>
            {message.role === "assistant" && <span className="avatar"><Bot size={16} aria-hidden="true" /></span>}
            <div>
              <p>{message.content}</p>
            </div>
            {message.role === "assistant" && (
              <div className="message-actions">
                <button onClick={() => onOpenPopup({ type: "source", message })}><IconLabel icon={Layers}>Sources</IconLabel></button>
                <button type="button"><IconLabel icon={Copy}>Copy</IconLabel></button>
                <button onClick={() => onOpenPopup({ type: "trace", message })}><IconLabel icon={GitBranch}>Trace</IconLabel></button>
                <button onClick={() => onFeedback(message, "up")}><IconLabel icon={ThumbsUp}>Useful</IconLabel></button>
                <button onClick={() => onFeedback(message, "down")}><IconLabel icon={ThumbsDown}>Needs work</IconLabel></button>
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
          <span><IconLabel icon={Paperclip}>Attach</IconLabel></span>
          <span><IconLabel icon={Filter}>Filter</IconLabel></span>
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
          <strong>{selectedRoute}</strong>
          <button className="send-action" onClick={onSend} disabled={isLoading || (requiresKnowledgeBase && !selectedKnowledgeBaseId)} aria-label="Send message"><SendHorizontal size={20} aria-hidden="true" /></button>
        </div>
      </div>
      {requiresKnowledgeBase && !selectedKnowledgeBaseId && (
        <p className="ai-disclaimer">Select a knowledge base to run Adaptive or L2 Simple RAG.</p>
      )}
      <p className="ai-disclaimer">This is an AI-powered system. Please verify answers since AI can make mistakes.</p>
    </section>
  );
}

function RagConfiguration({
  config,
  setConfig,
  selectedKnowledgeBase,
  lastAnswerMetadata,
  chatConfigurations = [],
  configurationStatus = "",
  onSelectChatConfiguration = () => {},
  onSaveConfiguration = () => {},
  onUpdateConfiguration = () => {},
  onResetConfiguration = () => {},
  collapsed,
  onToggle
}) {
  const providerModels = generatorModelOptionsByProvider[config.generatorProvider] || generatorModelOptionsByProvider.Local;
  if (collapsed) {
    return (
      <aside className="config-panel panel-rail config-rail">
        <button className="panel-rail-button" type="button" onClick={onToggle} aria-label="Expand configuration">
          <span>Config</span>
        </button>
      </aside>
    );
  }
  return (
    <aside className="config-panel">
      <div className="config-topbar">
        <div>
          <p className="eyebrow">Runtime</p>
          <h2>Configuration</h2>
        </div>
        <button className="panel-collapse-button" type="button" onClick={onToggle} aria-label="Collapse configuration"><IconOnly icon={ChevronRight} /></button>
      </div>
      <CurrentRouteSummary config={config} selectedKnowledgeBase={selectedKnowledgeBase} metadata={lastAnswerMetadata} />
      <section className="config-section runtime-section">
        <header>
          <h3>Adaptive RAG inputs</h3>
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
      </section>
      <section className="config-section runtime-section">
        <header>
          <div>
            <h3>Generator target & prompts</h3>
            <small>Supported local models execute at runtime. External providers are visible for roadmap clarity.</small>
          </div>
        </header>
        <SelectField
          label="Saved configuration"
          value={config.chatConfigurationId}
          options={[{ value: "", label: "Draft configuration" }, ...chatConfigurations.map((item) => ({ value: item.id, label: item.name }))]}
          onChange={onSelectChatConfiguration}
        />
        <label>
          Configuration name
          <input
            value={config.configurationName}
            onChange={(event) => setConfig({ ...config, configurationName: event.target.value, chatConfigurationId: "" })}
            placeholder="Workflow support assistant"
          />
        </label>
        <label>
          Description
          <input
            value={config.configurationDescription}
            onChange={(event) => setConfig({ ...config, configurationDescription: event.target.value })}
            placeholder="Short purpose for this preset"
          />
        </label>
        <div className="config-two-column">
          <SelectField
            label="Generator provider"
            value={config.generatorProvider}
            options={generatorProviderOptions}
            onChange={(generatorProvider) => setConfig({
              ...config,
              generatorProvider,
              generatorModel: (generatorModelsByProvider[generatorProvider] || generatorModelsByProvider.Local)[0]
            })}
          />
          <SelectField
            label="Generator model"
            value={config.generatorModel}
            options={providerModels}
            onChange={(generatorModel) => setConfig({ ...config, generatorModel })}
          />
        </div>
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
        <div className="config-actions-row">
          <button className="secondary-action" type="button" onClick={onResetConfiguration}><IconLabel icon={RefreshCw}>Reset</IconLabel></button>
          <button className="secondary-action" type="button" onClick={onSaveConfiguration}><IconLabel icon={Plus}>Save as new</IconLabel></button>
          <button className="primary-action" type="button" onClick={onUpdateConfiguration}><IconLabel icon={Save}>Update selected</IconLabel></button>
        </div>
        {configurationStatus && <p className="config-status-note">{configurationStatus}</p>}
      </section>
    </aside>
  );
}
function CurrentRouteSummary({ config, selectedKnowledgeBase, metadata }) {
  const routeLabel = metadata?.route_label || config.route;
  const routeLevel = metadata?.route_level || answerModeFromRoute(config.route);
  return (
    <section className="route-summary-card">
      <header>
        <div>
          <p className="eyebrow">Current route</p>
          <h3>{routeLabel}</h3>
        </div>
        <span className={`status-pill ${metadata?.retrieval_used ? "status-completed" : ""}`}>
          {metadata?.retrieval_used ? "Retrieval on" : "No retrieval"}
        </span>
      </header>
      <dl className="route-summary-list">
        <div>
          <dt>Mode</dt>
          <dd>{routeLevel}</dd>
        </div>
        <div>
          <dt>Knowledge base</dt>
          <dd>{selectedKnowledgeBase ? selectedKnowledgeBase.name : "Not selected"}</dd>
        </div>
        <div>
          <dt>Documents / chunks</dt>
          <dd>{selectedKnowledgeBase ? `${selectedKnowledgeBase.document_count} / ${selectedKnowledgeBase.chunk_count}` : "-"}</dd>
        </div>
        <div>
          <dt>Query embedding</dt>
          <dd>{selectedKnowledgeBase?.embedding_model || "Select a KB"}</dd>
        </div>
        <div>
          <dt>Complexity</dt>
          <dd>{metadata?.complexity_label || "Waiting for answer"}</dd>
        </div>
        <div>
          <dt>Retrieval</dt>
          <dd>{metadata?.retrieval_mode || retrievalModeValue(config.retrievalMode)}</dd>
        </div>
        <div>
          <dt>Latency</dt>
          <dd>{metadata?.latency_ms ? `${metadata.latency_ms} ms` : "-"}</dd>
        </div>
      </dl>
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
              <button className="secondary-action" onClick={onOpenDetail}><IconLabel icon={GitBranch}>Open RAGXplain detail</IconLabel></button>
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
        <button className={tab === "tokens" ? "active" : ""} onClick={() => setTab("tokens")}><IconLabel icon={BarChart3}>Token statistics</IconLabel></button>
        <button className={tab === "feedback" ? "active" : ""} onClick={() => setTab("feedback")}><IconLabel icon={ThumbsUp}>Detailed Statistics & Feedbacks</IconLabel></button>
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

function IconLabel({ icon: Icon, children, size = 16 }) {
  return (
    <span className="icon-label">
      <Icon className="button-icon" size={size} aria-hidden="true" strokeWidth={2} />
      <span>{children}</span>
    </span>
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
          <option key={option.value} value={option.value} disabled={option.disabled}>{option.label}</option>
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
        metadata: record.metadata || {}
      };
    }
    return {
      id: record.id,
      role: "assistant",
      question: record.metadata?.question || previousUserQuestion,
      content: record.content,
      contexts: record.contexts || [],
      metadata: record.metadata || {}
    };
  });
}

function applyChatConfigurationToDraft(current, record) {
  return applyChatConfigurationSnapshotToDraft(current, chatConfigurationSnapshotFromRecord(record), record.id);
}

function applyChatConfigurationSnapshotToDraft(current, snapshot = {}, configurationId = "") {
  const provider = generatorProviders.includes(snapshot.generator_provider) ? snapshot.generator_provider : defaultChatConfigurationDraft.generatorProvider;
  const models = generatorModelsByProvider[provider] || generatorModelsByProvider.Local;
  const model = models.includes(snapshot.generator_model) ? snapshot.generator_model : models[0];
  return {
    ...current,
    chatConfigurationId: configurationId || snapshot.id || "",
    configurationName: snapshot.name || defaultChatConfigurationDraft.configurationName,
    configurationDescription: snapshot.description || "",
    generatorProvider: provider,
    generatorModel: model,
    responseStructure: responseStructures.includes(snapshot.response_structure) ? snapshot.response_structure : defaultChatConfigurationDraft.responseStructure,
    tone: chatbotTones.includes(snapshot.tone) ? snapshot.tone : defaultChatConfigurationDraft.tone,
    humorLevel: clampNumber(snapshot.humor_level, defaultChatConfigurationDraft.humorLevel, 0, 5),
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
    metadata: { runtime: "configuration-only", actual_generator: "extractive" }
  };
}

function isValidChatConfigurationDraft(config) {
  return Boolean(
    (config.chatConfigurationId || config.configurationName?.trim()) &&
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
