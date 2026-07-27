const API_BASE_URL = import.meta.env.VITE_ARAGBIZ_API_URL || "http://127.0.0.1:8000";
const AUTH_STORAGE_KEY = "aragbiz:auth-token";

function authHeaders(extra = {}) {
  const token = window.localStorage.getItem(AUTH_STORAGE_KEY);
  return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
}

export function setAuthToken(token) {
  if (token) window.localStorage.setItem(AUTH_STORAGE_KEY, token);
  else window.localStorage.removeItem(AUTH_STORAGE_KEY);
}

export function hasAuthToken() {
  return Boolean(window.localStorage.getItem(AUTH_STORAGE_KEY));
}

export function clearAuthToken() {
  setAuthToken("");
}

export async function getCurrentUser() {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    headers: authHeaders()
  });
  await assertOk(response, "Authentication check failed");
  return response.json();
}

export async function updateCurrentUser(payload) {
  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Profile update failed");
  const result = await response.json();
  setAuthToken(result.access_token);
  return result;
}

export async function login(payload) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Login failed");
  const result = await response.json();
  setAuthToken(result.access_token);
  return result;
}

export async function signup(payload) {
  const response = await fetch(`${API_BASE_URL}/auth/signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Sign up failed");
  const result = await response.json();
  setAuthToken(result.access_token);
  return result;
}

export async function askQuestion(question, options = {}) {
  const response = await fetch(`${API_BASE_URL}/answer`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      question,
      conversation_id: options.conversationId || null,
      knowledge_base_id: options.knowledgeBaseId || null,
      document_ids: options.documentIds || [],
      mode: options.mode || "adaptive",
      retrieval_mode: options.retrievalMode || "hybrid",
      top_k: options.topK || 4,
      chat_configuration_id: options.chatConfigurationId || null,
      chat_configuration: options.chatConfiguration || null
    })
  });
  await assertOk(response, "Answer request failed");
  return response.json();
}

export async function askQuestionStream(question, options = {}, onEvent = () => {}) {
  const response = await fetch(`${API_BASE_URL}/answer/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    signal: options.signal,
    body: JSON.stringify({
      question,
      request_id: options.requestId || "",
      conversation_id: options.conversationId || null,
      knowledge_base_id: options.knowledgeBaseId || null,
      document_ids: options.documentIds || [],
      mode: options.mode || "adaptive",
      retrieval_mode: options.retrievalMode || "hybrid",
      top_k: options.topK || 4,
      chat_configuration_id: options.chatConfigurationId || null,
      chat_configuration: options.chatConfiguration || null
    })
  });
  await assertOk(response, "Streaming answer request failed");
  return consumeSseResponse(response, onEvent);
}

export async function regenerateAnswerStream(messageId, options = {}, onEvent = () => {}) {
  const response = await fetch(`${API_BASE_URL}/chat/messages/${encodeURIComponent(messageId)}/regenerate/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    signal: options.signal,
    body: JSON.stringify({
      request_id: options.requestId || "",
      knowledge_base_id: options.knowledgeBaseId || null,
      document_ids: options.documentIds || [],
      mode: options.mode || "adaptive",
      retrieval_mode: options.retrievalMode || "hybrid",
      top_k: options.topK || 4,
      chat_configuration_id: options.chatConfigurationId || null,
      chat_configuration: options.chatConfiguration || null
    })
  });
  await assertOk(response, "Regenerate answer failed");
  return consumeSseResponse(response, onEvent);
}

export async function retryAnswerStream(messageId, options = {}, onEvent = () => {}) {
  const response = await fetch(`${API_BASE_URL}/chat/messages/${encodeURIComponent(messageId)}/retry/stream`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    signal: options.signal,
    body: JSON.stringify({
      request_id: options.requestId || "",
      knowledge_base_id: options.knowledgeBaseId || null,
      document_ids: options.documentIds || [],
      mode: options.mode || "adaptive",
      retrieval_mode: options.retrievalMode || "hybrid",
      top_k: options.topK || 4,
      chat_configuration_id: options.chatConfigurationId || null,
      chat_configuration: options.chatConfiguration || null
    })
  });
  await assertOk(response, "Retry answer failed");
  return consumeSseResponse(response, onEvent);
}

export async function cancelAnswerRequest(requestId) {
  const response = await fetch(`${API_BASE_URL}/answer/requests/${encodeURIComponent(requestId)}/cancel`, {
    method: "POST",
    headers: authHeaders()
  });
  await assertOk(response, "Stop answer request failed");
  return response.json();
}

export async function listChatMessageVersions(messageId) {
  const response = await fetch(`${API_BASE_URL}/chat/messages/${encodeURIComponent(messageId)}/versions`, {
    headers: authHeaders()
  });
  await assertOk(response, "Answer versions request failed");
  return response.json();
}

export async function getTrace(traceId) {
  const response = await fetch(`${API_BASE_URL}/traces/${encodeURIComponent(traceId)}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Trace request failed");
  return response.json();
}

export async function getChatMessageTrace(messageId, versionNumber = null) {
  const params = new URLSearchParams();
  if (versionNumber) params.set("version_number", String(versionNumber));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/chat/messages/${encodeURIComponent(messageId)}/trace${suffix}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Message trace request failed");
  return response.json();
}

export async function downloadTrace(traceId) {
  const response = await fetch(`${API_BASE_URL}/traces/${encodeURIComponent(traceId)}/download`, {
    headers: authHeaders()
  });
  await assertOk(response, "Trace download failed");
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${traceId}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

async function consumeSseResponse(response, onEvent) {
  if (!response.body) throw new Error("Streaming is not supported by this browser.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completedPayload = null;
  let cancelledPayload = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split(/\n\n/);
    buffer = parts.pop() || "";
    for (const part of parts) {
      const event = parseSseEvent(part);
      if (!event) continue;
      onEvent(event);
      if (event.type === "completed") completedPayload = event.data;
      if (event.type === "cancelled") cancelledPayload = event.data;
      if (event.type === "error") {
        throw new Error(event.data?.detail || "Streaming answer failed.");
      }
    }
  }

  if (buffer.trim()) {
    const event = parseSseEvent(buffer);
    if (event) {
      onEvent(event);
      if (event.type === "completed") completedPayload = event.data;
      if (event.type === "cancelled") cancelledPayload = event.data;
      if (event.type === "error") throw new Error(event.data?.detail || "Streaming answer failed.");
    }
  }
  return completedPayload || cancelledPayload;
}

function parseSseEvent(raw) {
  const lines = raw.split(/\r?\n/);
  const typeLine = lines.find((line) => line.startsWith("event:"));
  const dataLines = lines.filter((line) => line.startsWith("data:"));
  if (!typeLine || dataLines.length === 0) return null;
  const type = typeLine.slice("event:".length).trim();
  const dataRaw = dataLines.map((line) => line.slice("data:".length).trimStart()).join("\n");
  try {
    return { type, data: JSON.parse(dataRaw) };
  } catch {
    return { type, data: dataRaw };
  }
}

export async function listChatConversations({ query = "", section = "" } = {}) {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  if (section) params.set("section", section);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/chat/conversations${suffix}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Chat conversations request failed");
  return response.json();
}

export async function createChatConversation(payload = {}) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create chat conversation failed");
  return response.json();
}

export async function updateChatConversation(conversationId, payload) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update chat conversation failed");
  return response.json();
}

export async function deleteChatConversation(conversationId) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete chat conversation failed");
  return response.json();
}

export async function listChatMessages(conversationId) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}/messages`, {
    headers: authHeaders()
  });
  await assertOk(response, "Chat messages request failed");
  return response.json();
}

export async function listChatConfigurations() {
  const response = await fetch(`${API_BASE_URL}/chat/configurations`, {
    headers: authHeaders()
  });
  await assertOk(response, "Chat configurations request failed");
  return response.json();
}

export async function getChatConfigurationLimits() {
  const response = await fetch(`${API_BASE_URL}/chat/configuration-limits`, {
    headers: authHeaders()
  });
  await assertOk(response, "Chat configuration limits request failed");
  return response.json();
}

export async function createChatConfiguration(payload) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create chat configuration failed");
  return response.json();
}

export async function updateChatConfiguration(configurationId, payload) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations/${configurationId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update chat configuration failed");
  return response.json();
}

export async function deleteChatConfiguration(configurationId) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations/${configurationId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete chat configuration failed");
  return response.json();
}

export async function listModelProviders() {
  const response = await fetch(`${API_BASE_URL}/model-farm/providers`, {
    headers: authHeaders()
  });
  await assertOk(response, "Model providers request failed");
  return response.json();
}

export async function listModelDeployments({ capability = "", enabled = "" } = {}) {
  const params = new URLSearchParams();
  if (capability) params.set("capability", capability);
  if (enabled !== "") params.set("enabled", String(enabled));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments${suffix}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Model deployments request failed");
  return response.json();
}

export async function listAgentTools(publicWebEnabled = false) {
  const params = new URLSearchParams({ public_web_enabled: String(Boolean(publicWebEnabled)) });
  const response = await fetch(`${API_BASE_URL}/rag/agent-tools?${params.toString()}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Agent tools request failed");
  return response.json();
}

export async function listModelConnections({ provider = "", enabled = "" } = {}) {
  const params = new URLSearchParams();
  if (provider) params.set("provider", provider);
  if (enabled !== "") params.set("enabled", String(enabled));
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/model-farm/connections${suffix}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Model connections request failed");
  return response.json();
}

export async function createModelConnection(payload) {
  const response = await fetch(`${API_BASE_URL}/model-farm/connections`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create model connection failed");
  return response.json();
}

export async function updateModelConnection(connectionId, payload) {
  const response = await fetch(`${API_BASE_URL}/model-farm/connections/${connectionId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update model connection failed");
  return response.json();
}

export async function deleteModelConnection(connectionId) {
  const response = await fetch(`${API_BASE_URL}/model-farm/connections/${connectionId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete model connection failed");
  return response.json();
}

export async function testModelConnection(connectionId) {
  const response = await fetch(`${API_BASE_URL}/model-farm/connections/${connectionId}/test`, {
    method: "POST",
    headers: authHeaders()
  });
  await assertOk(response, "Test model connection failed");
  return response.json();
}

export async function listConnectionModels(connectionId) {
  const response = await fetch(`${API_BASE_URL}/model-farm/connections/${connectionId}/available-models`, {
    headers: authHeaders()
  });
  await assertOk(response, "Available models request failed");
  return response.json();
}

export async function createModelDeploymentFromTemplate(payload) {
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments/from-template`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create model deployment failed");
  return response.json();
}

export async function updateModelDeployment(deploymentId, payload) {
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments/${deploymentId}`, {
    method: "PATCH",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update model deployment failed");
  return response.json();
}

export async function testModelDeploymentDraft(payload) {
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments/test-draft`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Test endpoint failed");
  return response.json();
}

export async function deleteModelDeployment(deploymentId) {
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments/${deploymentId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete model deployment failed");
  return response.json();
}

export async function testModelDeployment(deploymentId) {
  const response = await fetch(`${API_BASE_URL}/model-farm/deployments/${deploymentId}/test`, {
    method: "POST",
    headers: authHeaders()
  });
  await assertOk(response, "Test model deployment failed");
  return response.json();
}

export async function listModelUsage({ deploymentId = "", purpose = "", limit = 500 } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (deploymentId) params.set("deployment_id", deploymentId);
  if (purpose) params.set("purpose", purpose);
  const response = await fetch(`${API_BASE_URL}/model-farm/usage?${params.toString()}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Model usage request failed");
  return response.json();
}

export async function getModelUsageSummary() {
  const response = await fetch(`${API_BASE_URL}/model-farm/usage/summary`, {
    headers: authHeaders()
  });
  await assertOk(response, "Model usage summary request failed");
  return response.json();
}

export async function getAnalyticsOverview(filters = {}) {
  return getAnalyticsResource("/analytics/overview", filters, "Analytics overview request failed");
}

export async function getAnalyticsUsageTrend(filters = {}) {
  return getAnalyticsResource("/analytics/usage/trend", filters, "Analytics trend request failed");
}

export async function getAnalyticsUsageBreakdowns(filters = {}) {
  return getAnalyticsResource("/analytics/usage/breakdowns", filters, "Analytics breakdown request failed");
}

export async function listAnalyticsUsageEvents(filters = {}) {
  return getAnalyticsResource("/analytics/usage/events", filters, "Analytics usage request failed");
}

export async function listAnalyticsFeedback(filters = {}) {
  return getAnalyticsResource("/analytics/feedback", filters, "Analytics feedback request failed");
}

export async function getAnalyticsFilterOptions(filters = {}) {
  return getAnalyticsResource("/analytics/filter-options", filters, "Analytics filter request failed");
}

async function getAnalyticsResource(path, filters, errorLabel) {
  const params = analyticsSearchParams(filters);
  const response = await fetch(`${API_BASE_URL}${path}?${params.toString()}`, {
    headers: authHeaders()
  });
  await assertOk(response, errorLabel);
  return response.json();
}

function analyticsSearchParams(filters = {}) {
  const params = new URLSearchParams();
  const mapping = {
    from: "from",
    to: "to",
    scope: "scope",
    deploymentId: "deployment_id",
    knowledgeBaseId: "knowledge_base_id",
    chatConfigurationId: "chat_configuration_id",
    purpose: "purpose",
    status: "status",
    userId: "user_id",
    rating: "rating",
    query: "query",
    page: "page",
    pageSize: "page_size",
    metric: "metric"
  };
  Object.entries(mapping).forEach(([key, parameter]) => {
    const value = filters[key];
    if (value !== undefined && value !== null && value !== "") {
      params.set(parameter, String(value));
    }
  });
  return params;
}

export async function listJobs({ status = "", limit = 100 } = {}) {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.set("status", status);
  const response = await fetch(`${API_BASE_URL}/jobs?${params.toString()}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Jobs request failed");
  return response.json();
}

export async function getJob(jobId) {
  const response = await fetch(`${API_BASE_URL}/jobs/${jobId}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Job request failed");
  return response.json();
}

export async function listEvaluationRuns() {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation runs request failed");
  return response.json();
}

export async function listEvaluationDatasets(knowledgeBaseId = "") {
  const params = new URLSearchParams();
  if (knowledgeBaseId) params.set("knowledge_base_id", knowledgeBaseId);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/evaluation/datasets${suffix}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation datasets request failed");
  return response.json();
}

export async function listEvaluationExperiments() {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation experiments request failed");
  return response.json();
}

export async function createEvaluationExperiment(payload) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create evaluation experiment failed");
  return response.json();
}

export async function getEvaluationExperiment(experimentId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments/${experimentId}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation experiment request failed");
  return response.json();
}

export async function listEvaluationExperimentRuns(experimentId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments/${experimentId}/runs`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation experiment runs request failed");
  return response.json();
}

export async function cancelEvaluationExperiment(experimentId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments/${experimentId}/cancel`, {
    method: "POST",
    headers: authHeaders()
  });
  await assertOk(response, "Cancel evaluation experiment failed");
  return response.json();
}

export async function resumeEvaluationExperiment(experimentId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments/${experimentId}/resume`, {
    method: "POST",
    headers: authHeaders()
  });
  await assertOk(response, "Resume evaluation experiment failed");
  return response.json();
}

export async function deleteEvaluationExperiment(experimentId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/experiments/${experimentId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete evaluation experiment failed");
  return response.json();
}

export async function startRagxplainDiagnosis(runId, payload) {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs/${runId}/ragxplain`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Start RAGXplain diagnosis failed");
  return response.json();
}

export async function createEvaluationRun(payload) {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create evaluation run failed");
  return response.json();
}

export async function getEvaluationRun(runId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs/${runId}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation run request failed");
  return response.json();
}

export async function listEvaluationCases(runId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs/${runId}/cases`, {
    headers: authHeaders()
  });
  await assertOk(response, "Evaluation cases request failed");
  return response.json();
}

export async function deleteEvaluationRun(runId) {
  const response = await fetch(`${API_BASE_URL}/evaluation/runs/${runId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete evaluation run failed");
  return response.json();
}

export function getRagxplainViewerUrl(runId) {
  const encodedRunId = encodeURIComponent(runId);
  const insightsPath = `/evaluation/runs/${encodedRunId}/ragxplain/overall-insights`;
  return `${API_BASE_URL}/evaluation/ragxplain/viewer?insights=${encodeURIComponent(insightsPath)}`;
}
export async function submitFeedback(assistantMessageId, payload) {
  const response = await fetch(`${API_BASE_URL}/feedback/messages/${encodeURIComponent(assistantMessageId)}`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Feedback request failed");
  return response.json();
}

export async function listKnowledgeBases() {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`, {
    headers: authHeaders()
  });
  await assertOk(response, "Knowledge base request failed");
  return response.json();
}

export async function listKnowledgeSourceCatalog() {
  const response = await fetch(`${API_BASE_URL}/knowledge-sources/catalog`, {
    headers: authHeaders()
  });
  await assertOk(response, "Knowledge source catalog request failed");
  return response.json();
}

export async function createKnowledgeBase(payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create knowledge base failed");
  return response.json();
}

export async function updateKnowledgeBase(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update knowledge base failed");
  return response.json();
}

export async function deleteKnowledgeBase(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete knowledge base failed");
  return response.json();
}

export async function uploadKnowledgeSource(knowledgeBaseId, files) {
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/upload`, {
    method: "POST",
    headers: authHeaders(),
    body
  });
  await assertOk(response, "Upload source failed");
  return response.json();
}

export async function ingestWebsiteSource(knowledgeBaseId, url) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/website`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ url })
  });
  await assertOk(response, "Website ingestion failed");
  return response.json();
}

export async function importWixqaCorpus(
  knowledgeBaseId,
  confirmRemoteEmbedding = false,
  documentLimit = 6221
) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/wixqa-corpus`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      confirm_remote_embedding: Boolean(confirmRemoteEmbedding),
      document_limit: Number(documentLimit)
    })
  });
  await assertOk(response, "WixQA corpus import failed");
  return response.json();
}

export async function reindexKnowledgeBase(knowledgeBaseId, wait = true) {
  const response = await fetch(
    `${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/reindex?wait=${wait ? "true" : "false"}`,
    {
    method: "POST",
    headers: authHeaders()
    }
  );
  await assertOk(response, "Reindex failed");
  return response.json();
}

export async function listKnowledgeDocuments(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`, {
    headers: authHeaders()
  });
  await assertOk(response, "Documents request failed");
  return response.json();
}

export async function listKnowledgeDocumentsPage(knowledgeBaseId, { query = "", offset = 0, limit = 25 } = {}) {
  const params = new URLSearchParams({
    query,
    offset: String(offset),
    limit: String(limit)
  });
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents-page?${params.toString()}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Document page request failed");
  return response.json();
}

export async function listKnowledgeDocumentChunks(knowledgeBaseId, documentId) {
  const response = await fetch(
    `${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}/chunks`,
    { headers: authHeaders() }
  );
  await assertOk(response, "Document chunks request failed");
  return response.json();
}

export async function listKnowledgeChunks(knowledgeBaseId, limit = 1000) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/chunks?limit=${limit}`, {
    headers: authHeaders()
  });
  await assertOk(response, "Chunks request failed");
  return response.json();
}

export async function getKnowledgeProcessingTrace(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/processing-trace`, {
    headers: authHeaders()
  });
  await assertOk(response, "Processing trace request failed");
  return response.json();
}

export async function listKnowledgeIndexVersions(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/index-versions`, {
    headers: authHeaders()
  });
  await assertOk(response, "Index versions request failed");
  return response.json();
}

export async function createKnowledgeDocument(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create document failed");
  return response.json();
}

export async function updateKnowledgeDocument(knowledgeBaseId, documentId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "PUT",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update document failed");
  return response.json();
}

export async function deleteKnowledgeDocument(knowledgeBaseId, documentId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "DELETE",
    headers: authHeaders()
  });
  await assertOk(response, "Delete document failed");
  return response.json();
}

async function assertOk(response, fallbackMessage) {
  if (response.ok) return;
  const detail = await responseDetail(response);
  throw new Error(`${fallbackMessage}: ${response.status}${detail ? ` - ${detail}` : ""}`);
}

async function responseDetail(response) {
  const contentType = response.headers.get("content-type") || "";
  try {
    if (contentType.includes("application/json")) {
      const payload = await response.json();
      if (typeof payload.detail === "string") return payload.detail;
      if (Array.isArray(payload.detail)) return payload.detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
      return payload.message || payload.error || "";
    }
    return await response.text();
  } catch {
    return "";
  }
}
