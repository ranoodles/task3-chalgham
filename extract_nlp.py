import os
import re
from pathlib import Path
import csv
from typing import List, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv

#llamaindex imports
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.base.embeddings.base import similarity
import numpy as np

#load env vars
load_dotenv()

#init openai client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#the 5 goal queries for semantic search
GOAL_QUERIES = {
    "Goal 1": "Mechanisms by which humans interact with autonomous systems, including both workflow-level interaction and cognitive-flow processes",
    "Goal 2": "Novel characteristics and features of modern AI / autonomous systems in comparison with conventional automation used in high-risk industries",
    "Goal 3": "The types of human performance degradation that may arise from these new AI features",
    "Goal 4": "Measurable metrics for evaluating AI features that may degrade human performance during work, tasks, or operational activities, and explain how to use them",
    "Goal 5": "Current research to determine whether real, semi-real, or simulated human-AI interaction platforms or frameworks exist"
}


def split_large_chunk(chunk_text: str, max_chars: int = 30000) -> List[str]:
    if len(chunk_text) <= max_chars:
        return [chunk_text]
    
    sub_chunks = []
    
    #try splitting by paragraphs first
    paragraphs = chunk_text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        #if this para would exceed limit, save current and start new
        if len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
            sub_chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    #add final chunk
    if current_chunk.strip():
        sub_chunks.append(current_chunk.strip())
    
    #if still too large, split by sentences
    final_chunks = []
    for chunk in sub_chunks:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
        else:
            #split by sentences
            sentences = re.split(r'([.!?]\s+)', chunk)
            current = ""
            for i in range(0, len(sentences), 2):
                sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                if len(current) + len(sentence) > max_chars and current:
                    final_chunks.append(current.strip())
                    current = sentence
                else:
                    current += sentence
            if current.strip():
                final_chunks.append(current.strip())
    
    return final_chunks


def chunk_paper_semantically(text: str, embed_model: OpenAIEmbedding) -> List[Dict]:
    print("  Creating semantic chunks...")
    
    #create semantic splitter
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=embed_model
    )
    
    #create doc and split
    doc = Document(text=text)
    nodes = splitter.get_nodes_from_documents([doc])
    
    #convert nodes to dicts with embeddings
    chunks = []
    for i, node in enumerate(nodes):
        chunks.append({
            'id': i,
            'text': node.text,
            'embedding': node.embedding,
            'metadata': node.metadata
        })
    
    print(f"  Created {len(chunks)} semantic chunks")
    return chunks


def find_relevant_chunks(
    chunks: List[Dict],
    query_embeddings: Dict[str, List[float]],
    embed_model: OpenAIEmbedding,
    top_k: int = 7,
    similarity_threshold: float = 0.45
) -> Dict[str, List[Dict]]:
    print(f"  Finding top {top_k} relevant chunks for each goal (threshold: {similarity_threshold:.2f})...")
    
    relevant_chunks = {}
    all_similarities = []  #store all similarities for fallback
    
    #openai embedding models have 8192 token limit (~30k chars conservatively)
    MAX_EMBED_CHARS = 30000
    
    for goal_name, query_embedding in query_embeddings.items():
        #calc similarity scores for all chunks
        similarities = []
        for chunk in chunks:
            #get chunk embedding if not already present
            if chunk['embedding'] is None:
                chunk_text = chunk['text']
                
                #split oversized chunks before embedding
                if len(chunk_text) > MAX_EMBED_CHARS:
                    print(f"      ⚠ Chunk {chunk['id']} too large ({len(chunk_text):,} chars), splitting...")
                    sub_chunks = split_large_chunk(chunk_text, MAX_EMBED_CHARS)
                    print(f"        Split into {len(sub_chunks)} sub-chunks")
                    
                    #get embeddings for each sub-chunk and average them
                    sub_embeddings = []
                    for i, sub_chunk in enumerate(sub_chunks):
                        try:
                            sub_emb = embed_model.get_text_embedding(sub_chunk)
                            sub_embeddings.append(sub_emb)
                        except Exception as e:
                            print(f"        ✗ Error embedding sub-chunk {i+1}/{len(sub_chunks)}: {e}")
                    
                    if sub_embeddings:
                        #average the embeddings of sub-chunks
                        chunk['embedding'] = np.mean(sub_embeddings, axis=0).tolist()
                    else:
                        print(f"      ✗ Could not embed chunk {chunk['id']}, skipping")
                        continue
                else:
                    try:
                        chunk['embedding'] = embed_model.get_text_embedding(chunk_text)
                    except Exception as e:
                        print(f"      ✗ Error embedding chunk {chunk['id']}: {e}")
                        continue
            
            #calc cosine similarity using llamaindex's built-in function
            sim_score = similarity(query_embedding, chunk['embedding'])
            similarities.append((chunk, sim_score))
            #store for global fallback (with goal context)
            all_similarities.append((chunk, sim_score, goal_name))
        
        #filter by threshold and sort by similarity score (descending)
        filtered_similarities = [(chunk, score) for chunk, score in similarities 
                                if score >= similarity_threshold]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        #get top-k chunks (after filtering)
        top_chunks = [
            {
                'text': chunk['text'],
                'score': float(score),
                'chunk_id': chunk['id']
            }
            for chunk, score in filtered_similarities[:top_k]
        ]
        
        #only add goal if it has chunks above threshold
        if top_chunks:
            relevant_chunks[goal_name] = top_chunks
            avg_score = np.mean([c['score'] for c in top_chunks])
            print(f"    {goal_name}: avg similarity = {avg_score:.3f} ({len(top_chunks)} chunks)")
        else:
            print(f"    {goal_name}: No chunks above threshold (skipped)")
    
    #check if we have any chunks at all
    total_chunks = sum(len(chunks) for chunks in relevant_chunks.values())
    
    #if no chunks passed threshold, return top 2-3 globally as fallback
    if total_chunks == 0:
        print(f"  ⚠ All chunks filtered out! Using fallback: top chunks globally...")
        #sort all similarities globally and get top 2-3 chunks
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        fallback_count = min(3, len(all_similarities))  #up to 3 chunks, or fewer if not available
        top_global = all_similarities[:fallback_count]
        
        #create a special "fallback" goal entry
        relevant_chunks["_fallback"] = [
            {
                'text': chunk['text'],
                'score': float(score),
                'chunk_id': chunk['id'],
                'original_goal': goal_name
            }
            for chunk, score, goal_name in top_global
        ]
        avg_score = np.mean([c['score'] for c in relevant_chunks["_fallback"]])
        print(f"    Fallback: Selected {fallback_count} top chunks globally (avg similarity = {avg_score:.3f})")
    
    return relevant_chunks


def create_batch_extraction_prompt(
    paper_name: str,
    relevant_chunks: Dict[str, List[Dict]],
    goal_queries: Dict[str, str]
) -> str:
    prompt = f"""You are an expert systematic literature review coder specializing in AI-induced human performance degradation.

Your task is to analyze excerpts from the paper "{paper_name}" that were identified as relevant to human-AI interaction research. Extract ALL evidence that supports causal links between specific AI features and human performance degradation.

IMPORTANT: Look for ANY discussion of:
- How AI/automation affects human skills, abilities, or performance
- Problems that arise from human-AI interaction
- Negative consequences of AI features (even if not explicitly labeled as "degradation")
- Trade-offs where AI introduces new challenges for human users
- Studies showing performance differences with AI systems

Below are the relevant excerpts from the paper (sorted by relevance):

"""
    
    #deduplicate chunks by chunk_id (keep highest similarity score if duplicate)
    unique_chunks_dict = {}  #chunk_id -> chunk dict with highest score
    
    for goal_name, goal_chunks in relevant_chunks.items():
        for chunk in goal_chunks:
            chunk_id = chunk['chunk_id']
            #keep chunk with highest similarity score if duplicate
            if chunk_id not in unique_chunks_dict or chunk['score'] > unique_chunks_dict[chunk_id]['score']:
                unique_chunks_dict[chunk_id] = chunk
    
    #sort by similarity score (descending) and add to prompt
    sorted_chunks = sorted(unique_chunks_dict.values(), key=lambda x: x['score'], reverse=True)
    
    excerpt_counter = 1
    for chunk in sorted_chunks:
        prompt += f"[Excerpt {excerpt_counter}] (Similarity: {chunk['score']:.3f})\n"
        prompt += chunk['text']
        prompt += "\n\n"
        excerpt_counter += 1
    
    prompt += """
---

Now, extract ALL relevant evidence from these excerpts. Output ONLY in the following exact table format (TSV style, tab-separated). Do NOT add any extra text, explanations, or introductions outside the table.

Columns (exactly in this order):

Source | Your finding about AI features | Your finding about Human performance degradation types | Your finding about Causal links between them | Excerpt | Justification | Validation (y/n)

Rules:

- Source: Use "{paper_name}" for all rows.

- Your finding about AI features: List the specific AI feature(s) mentioned (e.g., "High-level automation", "Lack of explainability", "Over-trust prompts", "Opaque decision-making"). Use short, clear phrases.

- Your finding about Human performance degradation types: List the specific degradation type(s) mentioned (e.g., "Complacency", "Skill decay", "Loss of situation awareness", "Automation bias") along with a brief 1-sentence description of what that represents.

- Your finding about Causal links between them: Write 2-3 sentences as a causal statement (e.g., "High-level automation leads to complacency because operators reduce monitoring").

- Excerpt: Exact verbatim quote from the text provided above (no paraphrasing, no ellipses unless in original). Reference which excerpt number it came from if helpful.

- Justification: Two sentences explaining WHY this excerpt shows a causal link between the AI feature and the degradation.

- Validation (y/n): Always write "y" (we will validate later; this is just a placeholder).

If there are multiple findings across the excerpts, output one row per distinct finding.

If there are NO relevant findings, output exactly one row with "No relevant evidence found" in the Excerpt column and leave others blank.

Start output immediately with the header row:

Source\tYour finding about AI features\tYour finding about Human performance degradation types\tYour finding about Causal links between them\tExcerpt\tJustification\tValidation (y/n)
""".replace("{paper_name}", paper_name)
    
    return prompt


def extract_evidence_from_paper_nlp(
    paper_path: Path,
    embed_model: OpenAIEmbedding,
    query_embeddings: Dict[str, List[float]],
    top_k: int = 7,
    similarity_threshold: float = 0.45
) -> List[Dict]:
    paper_name = paper_path.stem
    
    print(f"\nProcessing: {paper_name}")
    
    #read the paper content
    try:
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  ✗ Error reading {paper_name}: {e}")
        return []
    
    print(f"  Paper length: {len(content):,} characters")
    
    #step 1: chunk the paper semantically
    try:
        chunks = chunk_paper_semantically(content, embed_model)
    except Exception as e:
        print(f"  ✗ Error chunking {paper_name}: {e}")
        return []
    
    #step 2: find relevant chunks for each goal query
    try:
        relevant_chunks = find_relevant_chunks(
            chunks, 
            query_embeddings, 
            embed_model, 
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    except Exception as e:
        print(f"  ✗ Error finding relevant chunks for {paper_name}: {e}")
        return []
    
    #count unique relevant chunks
    unique_chunks = set()
    for goal_name, goal_chunks in relevant_chunks.items():
        for chunk in goal_chunks:
            unique_chunks.add(chunk['chunk_id'])
    
    if "_fallback" in relevant_chunks:
        print(f"  Selected {len(unique_chunks)} unique chunks (fallback mode: top chunks globally)")
    else:
        print(f"  Selected {len(unique_chunks)} unique chunks (across all goals)")
    
    #step 3: create batch prompt with all relevant excerpts
    prompt = create_batch_extraction_prompt(paper_name, relevant_chunks, GOAL_QUERIES)
    
    #estimate token count (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(prompt) // 4
    print(f"  Prompt size: {len(prompt):,} chars (~{estimated_tokens:,} tokens)")
    
    #step 4: call openai api with the batch prompt
    try:
        print(f"  Sending to GPT-4o-mini... (this may take a moment)")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert systematic literature review coder. You extract evidence precisely and thoroughly. Look broadly for any evidence of AI impacts on human performance, even if subtle or indirect."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=6000
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"  ✓ Received response ({len(response_text)} characters)")
        
        #parse tsv response
        rows = parse_tsv_response(response_text, paper_name)
        print(f"  ✓ Extracted {len(rows)} finding(s)")
        
        return rows
        
    except Exception as e:
        print(f"  ✗ Error processing {paper_name} with GPT: {e}")
        return []


def parse_tsv_response(response_text: str, paper_name: str) -> List[Dict]:
    lines = response_text.strip().split('\n')
    
    if len(lines) < 2:  #need at least header and one data row
        print(f"  Warning: Unexpected response format for {paper_name}")
        return []
    
    #skip the header line (first line)
    data_lines = lines[1:]
    
    rows = []
    for line in data_lines:
        if not line.strip():
            continue
        
        #skip duplicate header rows
        if line.startswith('Source\t') or line.startswith('Source |'):
            continue
            
        #split by tab
        parts = line.split('\t')
        
        if len(parts) >= 7:
            #check if this is a "No relevant evidence found" row
            if "No relevant evidence found" in parts[4]:
                print(f"  Note: No relevant evidence found in this paper")
                continue
            
            row = {
                'Source': parts[0].strip(),
                'AI Features': parts[1].strip(),
                'Performance Degradation Types': parts[2].strip(),
                'Causal Links': parts[3].strip(),
                'Excerpt': parts[4].strip(),
                'Justification': parts[5].strip(),
                'Validation': parts[6].strip()
            }
            rows.append(row)
        else:
            print(f"  Warning: Skipping malformed row with {len(parts)} columns (expected 7)")
    
    return rows


def main():
    #set up paths
    base_dir = Path(__file__).parent
    parsed_files_dir = base_dir / 'parsedFilesLlama'
    output_file = base_dir / 'nlp_deg_evidence.csv'
    
    #get all text files
    text_files = sorted(parsed_files_dir.glob('*.txt'))
    
    if not text_files:
        print("No text files found in parsedFilesLlama directory!")
        return
    
    print("=" * 80)
    print("AI PERFORMANCE DEGRADATION EVIDENCE EXTRACTION")
    print("Using NLP Semantic Chunking + Vector Similarity Approach")
    print("=" * 80)
    print(f"\nFound {len(text_files)} text files to process")
    print(f"Output will be saved to: {output_file}")
    
    #init embedding model
    print("\nInitializing OpenAI embedding model...")
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv('OPENAI_API_KEY')
    )
    print("✓ Embedding model ready")
    
    #pre-compute embeddings for goal queries
    print("\nComputing embeddings for 5 research goal queries...")
    query_embeddings = {}
    for goal_name, query_text in GOAL_QUERIES.items():
        query_embeddings[goal_name] = embed_model.get_text_embedding(query_text)
        print(f"  ✓ {goal_name}")
    
    print("\n" + "=" * 80)
    print("PROCESSING PAPERS")
    print("=" * 80)
    
    #collect all extracted evidence
    all_evidence = []
    
    for i, text_file in enumerate(text_files, 1):
        print(f"\n[{i}/{len(text_files)}] {text_file.name}")
        print("-" * 80)
        
        evidence_rows = extract_evidence_from_paper_nlp(
            text_file,
            embed_model,
            query_embeddings,
            top_k=7,  # Top 7 chunks per goal
            similarity_threshold=0.45  # Only include chunks with similarity >= 0.45
        )
        all_evidence.extend(evidence_rows)
    
    #write to csv
    print("\n" + "=" * 80)
    if all_evidence:
        print(f"Writing {len(all_evidence)} total findings to CSV...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Source', 'AI Features', 'Performance Degradation Types', 
                         'Causal Links', 'Excerpt', 'Justification', 'Validation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(all_evidence)
        
        print(f"✓ Successfully wrote to: {output_file}")
        print(f"✓ Total evidence rows: {len(all_evidence)}")
        
        #print summary stats
        unique_papers = set(row['Source'] for row in all_evidence)
        print(f"✓ Evidence found in {len(unique_papers)} papers")
        print(f"✓ Average findings per paper: {len(all_evidence) / len(unique_papers):.1f}")
    else:
        print("\nNo evidence was extracted from any papers!")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)


main()