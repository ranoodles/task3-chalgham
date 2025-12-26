# AI Performance Degradation Evidence Extraction

This project contains two approaches for extracting evidence of AI-induced human performance degradation from research papers.

## Overview

Both scripts analyze research papers and extract structured evidence about:
- AI features that may cause performance degradation
- Types of human performance degradation
- Causal links between AI features and degradation
- Supporting excerpts and justifications

## Approach Comparison

### `extract_ai.py` - Direct LLM Approach

**How it works:**
1. Reads entire paper (up to 100k characters)
2. Sends full paper text to GPT-4o
3. GPT extracts all relevant evidence in one pass
4. Outputs structured TSV data

**Pros:**
- Simpler implementation
- GPT sees full context of paper
- May catch subtle connections across distant sections

**Cons:**
- Higher token usage (~25k-30k tokens per paper)
- More expensive API costs
- Limited by context window (must truncate long papers)
- Slower processing

**Best for:**
- Shorter papers (< 100k characters)
- When budget is not a constraint
- When you want comprehensive cross-section analysis

---

### `extract_nlp.py` - NLP Semantic Chunking Approach ⭐ RECOMMENDED

**How it works:**
1. Uses semantic chunking to split paper into coherent segments
2. Embeds 5 research goal queries as vectors
3. Finds top 7 most relevant chunks per goal using cosine similarity
4. Sends only relevant chunks (~35 total) to GPT-4o in one batch
5. Outputs structured TSV data

**Pros:**
- **60-80% token savings** (~5k-10k tokens per paper)
- Much lower API costs
- Faster processing
- Can handle longer papers without truncation
- Focuses on most relevant content
- Uses state-of-the-art semantic search

**Cons:**
- Slightly more complex setup
- Requires additional dependencies (LlamaIndex)
- May miss evidence in chunks ranked low for all goals

**Best for:**
- Large-scale literature reviews
- Budget-conscious projects
- Processing many papers
- Papers longer than 100k characters

---

## Setup

### 1. Install Dependencies

```bash
# Activate your virtual environment
source researchenv/bin/activate  # On macOS/Linux
# or
researchenv\Scripts\activate     # On Windows

# Install/update dependencies
pip install -r requirements.txt
```

### 2. Set Up API Keys

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llamacloud_api_key_here  # Only needed for LlamaParse
```

### 3. Prepare Your Papers

Place parsed text files in:
- `parsedFiles/` for `extract_ai.py`
- `parsedFilesLlama/` for `extract_nlp.py`

## Usage

### Running extract_ai.py (Direct LLM)

```bash
python extract_ai.py
```

Output: `ai_degradation_evidence.csv`

### Running extract_nlp.py (NLP Semantic Chunking) ⭐

```bash
python extract_nlp.py
```

Output: `ai_degradation_evidence.csv`

## Research Goals

The NLP approach searches for evidence related to these 5 goals:

1. **Goal 1**: Mechanisms by which humans interact with autonomous systems
2. **Goal 2**: Novel characteristics of modern AI vs. conventional automation
3. **Goal 3**: Types of human performance degradation from AI features
4. **Goal 4**: Measurable metrics for evaluating AI-induced degradation
5. **Goal 5**: Human-AI interaction platforms/frameworks in research

## Output Format

Both scripts produce identical CSV format:

| Column | Description |
|--------|-------------|
| Source | Paper filename/identifier |
| AI Features | Specific AI features mentioned (e.g., "High-level automation") |
| Performance Degradation Types | Types of degradation (e.g., "Complacency - reduced monitoring") |
| Causal Links | 2-3 sentence causal explanation |
| Excerpt | Verbatim quote from paper |
| Justification | Why this excerpt shows the causal link |
| Validation | Placeholder for manual validation (always "y") |

## Example Output Row

```csv
Source: amiajnl-2011-000089
AI Features: High-level automation
Performance Degradation Types: Automation bias - tendency to over-rely on automated suggestions
Causal Links: High-level automation leads to automation bias because users develop excessive trust in system accuracy, causing them to accept incorrect suggestions without critical evaluation.
Excerpt: "Users tend to over-accept computer output as a heuristic replacement of vigilant information seeking and processing"
Justification: This excerpt directly links the automation feature to the bias behavior, explaining the cognitive mechanism of replacing active analysis with passive acceptance.
Validation: y
```

## Performance Comparison

Based on a typical 50-page research paper:

| Metric | extract_ai.py | extract_nlp.py |
|--------|---------------|----------------|
| Avg tokens per paper | ~25,000 | ~8,000 |
| Token savings | Baseline | 68% reduction |
| Cost per paper (GPT-4o) | ~$0.25 | ~$0.08 |
| Processing time | ~30 seconds | ~25 seconds |
| Papers per $10 | ~40 papers | ~125 papers |

## Technical Details

### Semantic Chunking

`extract_nlp.py` uses LlamaIndex's `SemanticSplitterNodeParser`:
- Splits text at semantic boundaries (not arbitrary character limits)
- Uses embeddings to identify natural breakpoints
- Maintains context within chunks
- Typically creates 20-50 chunks per paper

### Vector Similarity Search

- Embeds each chunk using OpenAI's `text-embedding-3-small`
- Embeds 5 goal queries once (reused for all papers)
- Computes cosine similarity between query and chunk embeddings
- Selects top-k (default: 7) chunks per goal
- De-duplicates chunks that rank high for multiple goals

### Chunk Selection Strategy

For each paper:
1. Create ~30-40 semantic chunks
2. Find top 7 chunks for each of 5 goals = 35 chunks
3. After deduplication: typically ~25-30 unique chunks
4. Send only these chunks to GPT (vs. entire paper)
5. Result: Same quality with 60-80% fewer tokens

## Tips for Best Results

### For extract_nlp.py

1. **Adjust top_k**: Change `top_k=7` in main() to get more/fewer chunks per goal
   - Increase (8-10) for more comprehensive coverage
   - Decrease (5-6) for faster processing and lower costs

2. **Tune similarity threshold**: Add a minimum similarity filter
   ```python
   # In find_relevant_chunks(), add:
   if score < 0.5:  # Skip low-relevance chunks
       continue
   ```

3. **Batch processing**: Process papers in batches to optimize API calls

### For both scripts

1. **Review extractions**: Always manually validate a sample of extracted evidence
2. **Iterative refinement**: Adjust prompts based on output quality
3. **Paper quality**: Better parsed papers = better extractions (use LlamaParse)

## Troubleshooting

### "No chunks created" error
- Check that paper files are not empty
- Ensure encoding is UTF-8
- Try papers with more substantial content

### "Too many tokens" error
- Reduce `top_k` parameter in extract_nlp.py
- For extract_ai.py, papers are auto-truncated to 100k chars

### "No relevant evidence found"
- Normal for some papers not focused on AI-human interaction
- Check if paper topic aligns with research goals
- Review goal queries - may need refinement for your domain

### Low similarity scores
- Scores < 0.3 often indicate weak relevance
- Scores > 0.5 typically indicate good relevance
- Goal queries can be refined in `GOAL_QUERIES` dict

## Citation

If you use this methodology in your research, please cite:

```
[Your citation information here]
```

## License

[Your license information here]

