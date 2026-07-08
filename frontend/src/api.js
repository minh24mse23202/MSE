const API_BASE_URL = import.meta.env.VITE_ARAGBIZ_API_URL || "http://127.0.0.1:8000";

export async function askQuestion(question, options = {}) {
  const response = await fetch(`${API_BASE_URL}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      conversation_id: options.conversationId || null,
      knowledge_base_id: options.knowledgeBaseId || null,
      mode: options.mode || "adaptive",
      retrieval_mode: options.retrievalMode || "hybrid",
      top_k: options.topK || 4
    })
  });
  await assertOk(response, "Answer request failed");
  return response.json();
}

export async function listChatConversations({ query = "", section = "" } = {}) {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  if (section) params.set("section", section);
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await fetch(`${API_BASE_URL}/chat/conversations${suffix}`);
  await assertOk(response, "Chat conversations request failed");
  return response.json();
}

export async function createChatConversation(payload = {}) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create chat conversation failed");
  return response.json();
}

export async function updateChatConversation(conversationId, payload) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update chat conversation failed");
  return response.json();
}

export async function deleteChatConversation(conversationId) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}`, {
    method: "DELETE"
  });
  await assertOk(response, "Delete chat conversation failed");
  return response.json();
}

export async function listChatMessages(conversationId) {
  const response = await fetch(`${API_BASE_URL}/chat/conversations/${conversationId}/messages`);
  await assertOk(response, "Chat messages request failed");
  return response.json();
}

export async function listChatConfigurations() {
  const response = await fetch(`${API_BASE_URL}/chat/configurations`);
  await assertOk(response, "Chat configurations request failed");
  return response.json();
}

export async function createChatConfiguration(payload) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create chat configuration failed");
  return response.json();
}

export async function updateChatConfiguration(configurationId, payload) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations/${configurationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update chat configuration failed");
  return response.json();
}

export async function deleteChatConfiguration(configurationId) {
  const response = await fetch(`${API_BASE_URL}/chat/configurations/${configurationId}`, {
    method: "DELETE"
  });
  await assertOk(response, "Delete chat configuration failed");
  return response.json();
}
export async function submitFeedback(payload) {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Feedback request failed");
  return response.json();
}

export async function listKnowledgeBases() {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`);
  await assertOk(response, "Knowledge base request failed");
  return response.json();
}

export async function createKnowledgeBase(payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create knowledge base failed");
  return response.json();
}

export async function updateKnowledgeBase(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update knowledge base failed");
  return response.json();
}

export async function deleteKnowledgeBase(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "DELETE"
  });
  await assertOk(response, "Delete knowledge base failed");
  return response.json();
}

export async function uploadKnowledgeSource(knowledgeBaseId, files) {
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/upload`, {
    method: "POST",
    body
  });
  await assertOk(response, "Upload source failed");
  return response.json();
}

export async function ingestWebsiteSource(knowledgeBaseId, url) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/website`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });
  await assertOk(response, "Website ingestion failed");
  return response.json();
}

export async function reindexKnowledgeBase(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/reindex`, {
    method: "POST"
  });
  await assertOk(response, "Reindex failed");
  return response.json();
}

export async function listKnowledgeDocuments(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`);
  await assertOk(response, "Documents request failed");
  return response.json();
}

export async function listKnowledgeChunks(knowledgeBaseId, limit = 1000) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/chunks?limit=${limit}`);
  await assertOk(response, "Chunks request failed");
  return response.json();
}

export async function getKnowledgeProcessingTrace(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/processing-trace`);
  await assertOk(response, "Processing trace request failed");
  return response.json();
}

export async function createKnowledgeDocument(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Create document failed");
  return response.json();
}

export async function updateKnowledgeDocument(knowledgeBaseId, documentId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  await assertOk(response, "Update document failed");
  return response.json();
}

export async function deleteKnowledgeDocument(knowledgeBaseId, documentId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "DELETE"
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
