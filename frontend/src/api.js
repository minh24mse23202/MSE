const API_BASE_URL = import.meta.env.VITE_ARAGBIZ_API_URL || "http://127.0.0.1:8000";

export async function askQuestion(question, knowledgeBaseId = "") {
  const response = await fetch(`${API_BASE_URL}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, knowledge_base_id: knowledgeBaseId || null })
  });
  if (!response.ok) {
    throw new Error(`Answer request failed: ${response.status}`);
  }
  return response.json();
}

export async function submitFeedback(payload) {
  const response = await fetch(`${API_BASE_URL}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Feedback request failed: ${response.status}`);
  }
  return response.json();
}

export async function listKnowledgeBases() {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`);
  if (!response.ok) {
    throw new Error(`Knowledge base request failed: ${response.status}`);
  }
  return response.json();
}

export async function createKnowledgeBase(payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Create knowledge base failed: ${response.status}`);
  }
  return response.json();
}

export async function updateKnowledgeBase(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Update knowledge base failed: ${response.status}`);
  }
  return response.json();
}

export async function deleteKnowledgeBase(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Delete knowledge base failed: ${response.status}`);
  }
  return response.json();
}

export async function uploadKnowledgeSource(knowledgeBaseId, files) {
  const body = new FormData();
  Array.from(files).forEach((file) => body.append("files", file));
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/upload`, {
    method: "POST",
    body
  });
  if (!response.ok) {
    throw new Error(`Upload source failed: ${response.status}`);
  }
  return response.json();
}

export async function ingestWebsiteSource(knowledgeBaseId, url) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/sources/website`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url })
  });
  if (!response.ok) {
    throw new Error(`Website ingestion failed: ${response.status}`);
  }
  return response.json();
}

export async function reindexKnowledgeBase(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/reindex`, {
    method: "POST"
  });
  if (!response.ok) {
    throw new Error(`Reindex failed: ${response.status}`);
  }
  return response.json();
}

export async function listKnowledgeDocuments(knowledgeBaseId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`);
  if (!response.ok) {
    throw new Error(`Documents request failed: ${response.status}`);
  }
  return response.json();
}

export async function createKnowledgeDocument(knowledgeBaseId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Create document failed: ${response.status}`);
  }
  return response.json();
}

export async function updateKnowledgeDocument(knowledgeBaseId, documentId, payload) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    throw new Error(`Update document failed: ${response.status}`);
  }
  return response.json();
}

export async function deleteKnowledgeDocument(knowledgeBaseId, documentId) {
  const response = await fetch(`${API_BASE_URL}/knowledge-bases/${knowledgeBaseId}/documents/${documentId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error(`Delete document failed: ${response.status}`);
  }
  return response.json();
}
