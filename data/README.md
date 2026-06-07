# Data

The full WixQA dataset is not committed to this repository. Download it from Hugging Face and convert it into local project formats:

```powershell
$env:PYTHONPATH='src'
python scripts/download_wixqa.py --subset wixqa_expertwritten
```

Generated files:

- `data/raw/wixqa/wix_kb_corpus.jsonl`
- `data/raw/wixqa/wixqa_expertwritten.jsonl`
- `data/processed/wix_kb_corpus_documents.jsonl`
- `data/processed/wixqa_expertwritten_qac.jsonl`

The Hugging Face dataset card describes WixQA as a MIT-licensed enterprise RAG QA benchmark with QA configs grounded in a Wix Help Center KB corpus.
