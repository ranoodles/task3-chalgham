"""
Test script for extract_nlp.py to verify functionality on a single paper.
This helps debug issues before running on the full corpus.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Import functions from extract_nlp
from extract_nlp import (
    chunk_paper_semantically,
    find_relevant_chunks,
    GOAL_QUERIES,
    extract_evidence_from_paper_nlp
)


def test_single_paper(paper_path: Path):
    """Test the extraction pipeline on a single paper."""
    print("=" * 80)
    print("TESTING EXTRACT_NLP.PY ON SINGLE PAPER")
    print("=" * 80)
    print(f"\nTest paper: {paper_path.name}")
    
    # Initialize embedding model
    print("\n1. Initializing embedding model...")
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv('OPENAI_API_KEY')
    )
    print("   ✓ Embedding model ready")
    
    # Read paper
    print("\n2. Reading paper...")
    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"   ✓ Paper length: {len(content):,} characters")
    
    # Test semantic chunking
    print("\n3. Testing semantic chunking...")
    try:
        chunks = chunk_paper_semantically(content, embed_model)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Show sample chunks
        print("\n   Sample chunks:")
        for i in [0, len(chunks)//2, -1]:
            chunk_preview = chunks[i]['text'][:100].replace('\n', ' ')
            print(f"   - Chunk {i}: {chunk_preview}...")
    except Exception as e:
        print(f"   ✗ Error during chunking: {e}")
        return
    
    # Test query embeddings
    print("\n4. Computing query embeddings...")
    query_embeddings = {}
    for goal_name, query_text in GOAL_QUERIES.items():
        query_embeddings[goal_name] = embed_model.get_text_embedding(query_text)
        print(f"   ✓ {goal_name}")
    
    # Test relevance finding
    print("\n5. Finding relevant chunks...")
    try:
        relevant_chunks = find_relevant_chunks(chunks, query_embeddings, embed_model, top_k=5)
        
        print("\n   Top relevant chunks per goal:")
        for goal_name, goal_chunks in relevant_chunks.items():
            print(f"\n   {goal_name}:")
            for i, chunk in enumerate(goal_chunks[:3], 1):  # Show top 3
                print(f"      {i}. Score: {chunk['score']:.3f}")
                preview = chunk['text'][:80].replace('\n', ' ')
                print(f"         {preview}...")
    except Exception as e:
        print(f"   ✗ Error finding relevant chunks: {e}")
        return
    
    # Test full extraction
    print("\n6. Testing full extraction pipeline...")
    try:
        evidence = extract_evidence_from_paper_nlp(
            paper_path,
            embed_model,
            query_embeddings,
            top_k=7
        )
        
        print(f"\n   ✓ Extracted {len(evidence)} findings")
        
        if evidence:
            print("\n   Sample finding:")
            sample = evidence[0]
            print(f"   - AI Feature: {sample['AI Features']}")
            print(f"   - Degradation: {sample['Performance Degradation Types'][:80]}...")
            print(f"   - Causal Link: {sample['Causal Links'][:100]}...")
        else:
            print("   Note: No evidence found in this paper")
            
    except Exception as e:
        print(f"   ✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nThe extract_nlp.py script is working correctly.")
    print("You can now run: python extract_nlp.py")


def main():
    """Run the test on a sample paper."""
    base_dir = Path(__file__).parent
    parsed_files_dir = base_dir / 'parsedFilesLlama'
    
    # Get first available text file
    text_files = list(parsed_files_dir.glob('*.txt'))
    
    if not text_files:
        print("ERROR: No text files found in parsedFilesLlama directory!")
        print(f"Please ensure papers are in: {parsed_files_dir}")
        return
    
    # Test on first paper
    test_paper = text_files[0]
    test_single_paper(test_paper)


if __name__ == "__main__":
    main()

