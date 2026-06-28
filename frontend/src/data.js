export const conversations = [
  {
    id: "conv-1024",
    title: "Wix Payments verification",
    updatedAt: "09:42",
    status: "active",
    route: "L2 Simple RAG"
  },
  {
    id: "conv-1021",
    title: "Invoice mismatch workflow",
    updatedAt: "Yesterday",
    status: "review",
    route: "L3 Complex RAG"
  },
  {
    id: "conv-1017",
    title: "Knowledge ingestion setup",
    updatedAt: "Mon",
    status: "archived",
    route: "L1 Direct"
  }
];

export const seedMessages = [
  {
    id: "m1",
    role: "assistant",
    content:
      "Welcome to Adaptive RAG Studio. Ask a workflow question and I will route it through the selected classifier, retriever, and generator path.",
    metadata: {
      complexity_label: "moderate",
      retrieval_mode: "hybrid",
      top_k: 4,
      latency_ms: 0
    },
    contexts: []
  }
];

export const knowledgeBases = [
  {
    id: "kb-wix",
    name: "Wix Help Center",
    source: "Wix/WixQA",
    documents: 6221,
    chunks: 18420,
    embeddingVersion: "v2026.06",
    status: "Indexed"
  },
  {
    id: "kb-procurement",
    name: "Procurement Workflow Samples",
    source: "Local JSONL",
    documents: 6,
    chunks: 19,
    embeddingVersion: "baseline",
    status: "Ready"
  },
  {
    id: "kb-hr",
    name: "HR Workflow Samples",
    source: "Local JSONL",
    documents: 3,
    chunks: 8,
    embeddingVersion: "baseline",
    status: "Ready"
  }
];

export const evaluationRuns = [
  {
    id: "eval-032",
    dataset: "WixQA expert-written",
    classifier: "DistilBERT",
    routingAccuracy: "0.80",
    contextRelevance: "0.50",
    faithfulness: "0.96",
    latency: "3.7s",
    status: "Latest"
  },
  {
    id: "eval-029",
    dataset: "WixQA + synthetic bootstrap",
    classifier: "Naive Bayes",
    routingAccuracy: "0.74",
    contextRelevance: "0.50",
    faithfulness: "0.95",
    latency: "0.8s",
    status: "Baseline"
  }
];

export const tokenStats = [
  { label: "Input tokens", value: "128.4k", delta: "+8.2%" },
  { label: "Output tokens", value: "42.7k", delta: "+4.1%" },
  { label: "Retrieval calls", value: "1,284", delta: "+12.5%" },
  { label: "Avg. cost proxy", value: "0.003", delta: "-2.0%" }
];

export const feedbackRows = [
  { user: "analyst@demo", rating: "up", topic: "Payments verification", note: "Citations were useful." },
  { user: "qa@demo", rating: "down", topic: "Invoice mismatch", note: "Needs clearer escalation step." },
  { user: "admin@demo", rating: "up", topic: "Knowledge base", note: "Trace view explains retrieval well." }
];

export const architectureLayers = [
  "Application layer: React RAG Studio, chatbot, analytics and evaluation",
  "Adaptive RAG: classifier, route decision, retriever, reranker, prompt assembly, generator",
  "Knowledge processing: connectors, loaders, chunking, embeddings, index management",
  "Storage: vector DB, relational DB, chat history, metadata, usage and audit logs",
  "Model farm: DistilBERT, T5-small, embedding models, rerankers and generator models",
  "Governance: RBAC, policy checks, citation validation and hallucination detection"
];
