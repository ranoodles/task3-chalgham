# Research Paper Evidence Extraction

## Purpose

Extracts evidence of causal links between AI features and human performance degradation from scientific papers for systematic literature review analysis. Outputs structured CSV files with findings about AI features, degradation types, causal links, excerpts, and justifications.

## Usage

### API Solution
```bash
python extract_ai.py
```
- Reads parsed text files from `parsedFiles/`
- Outputs to `ai_degradation_evidence_new.csv`

### NLP Solution
```bash
python extract_nlp.py
```
- Reads parsed text files from `parsedFilesLlama/`
- Outputs to `nlp_deg_evidence.csv`

## Technical Approaches

### API Solution (`extract_ai.py`)
- **Method**: Direct API extraction using GPT-4o-mini
- **PDF Usage**: Reads pre-parsed `.txt` files from `parsedFiles/`
- **Process**: Sends full paper content (truncated to 100k chars) directly to OpenAI API with structured extraction prompt
- **Advantages**: Simple, fast, processes entire paper at once

### NLP Solution (`extract_nlp.py`)
- **Method**: Semantic chunking + vector similarity filtering before API extraction
- **PDF Usage**: Reads pre-parsed `.txt` files from `parsedFilesLlama/`
- **Process**: 
  1. Semantically chunks papers using LlamaIndex SemanticSplitterNodeParser
  2. Embeds chunks and 5 research goal queries using OpenAI embeddings
  3. Filters top-k most relevant chunks per goal using cosine similarity (threshold: 0.45)
  4. Sends only relevant chunks to GPT-4o-mini for extraction
- **Advantages**: More targeted, reduces token usage, focuses on semantically relevant sections

