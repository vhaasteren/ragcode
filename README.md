# ragcode

A repo-aware RAG CLI to supercharge LLM-assisted coding. Designed to index large codebases
(PINT preset included), answer questions with citations, and export snippet packs for Cursor / IDEs.

## Quick start

```bash
pip install -e .
```

# default profile uses OpenAI embeddings
ragcode index --profile pint
ragcode query "simulate fake pulsar dataset ~1000 TOAs over 15yr" --k 8 --citations
ragcode dump --query "ModelBuilder parse .par and residuals" --k 12 --out cursor_context.md
ragcode serve --host 127.0.0.1 --port 8008

## Incremental indexing

```bash
# incremental (default)
ragcode index --profile pint --incremental

# force full rebuild
ragcode index --profile pint --no-incremental
```
