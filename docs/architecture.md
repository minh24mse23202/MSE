# Architecture

```mermaid
flowchart LR
    User["User Question"] --> Classifier["Query Complexity Classifier"]
    Classifier --> Router["Adaptive Router"]
    Router --> Retriever["Hybrid Retriever"]
    Retriever --> Generator["Generator"]
    Generator --> Answer["Answer + Contexts + Metadata"]
    Answer --> Feedback["Human Feedback Store"]
```

Core interfaces:

- `QueryClassifier.predict(query) -> complexity_label`
- `Retriever.search(query, top_k, mode) -> contexts`
- `AdaptiveRouter.route(query) -> rag_strategy`
- `RAGPipeline.answer(query) -> answer, contexts, metadata`
- `Evaluator.evaluate(dataset) -> metrics`

