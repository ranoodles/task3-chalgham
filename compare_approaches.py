"""
Utility script to compare token usage and costs between extract_ai.py and extract_nlp.py
Run this to estimate costs before processing your entire corpus.
"""

from pathlib import Path
from typing import Dict, List


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token for English text."""
    return len(text) // 4


def estimate_approach_costs(paper_paths: List[Path]) -> Dict:
    """Estimate costs for both approaches."""
    
    # GPT-4o pricing (as of 2024)
    INPUT_COST_PER_1M = 2.50  # $2.50 per 1M input tokens
    OUTPUT_COST_PER_1M = 10.00  # $10.00 per 1M output tokens
    EMBED_COST_PER_1M = 0.02  # $0.02 per 1M tokens for text-embedding-3-small
    
    # Average output tokens (based on typical extractions)
    AVG_OUTPUT_TOKENS = 2000
    
    results = {
        'ai': {'name': 'extract_ai.py (Direct LLM)', 'papers': [], 'total_input_tokens': 0, 'total_embed_tokens': 0},
        'nlp': {'name': 'extract_nlp.py (NLP Chunking)', 'papers': [], 'total_input_tokens': 0, 'total_embed_tokens': 0}
    }
    
    print("Analyzing papers...")
    print("=" * 80)
    
    for paper_path in paper_paths:
        with open(paper_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        paper_info = {
            'name': paper_path.name,
            'chars': len(content),
            'ai_input_tokens': 0,
            'nlp_input_tokens': 0,
            'nlp_embed_tokens': 0
        }
        
        # Approach 1: Direct LLM (extract_ai.py)
        # Truncate if over 100k chars
        ai_content = content[:100000] if len(content) > 100000 else content
        ai_tokens = estimate_tokens(ai_content)
        paper_info['ai_input_tokens'] = ai_tokens
        results['ai']['papers'].append(paper_info.copy())
        results['ai']['total_input_tokens'] += ai_tokens
        
        # Approach 2: NLP Chunking (extract_nlp.py)
        # Estimate: semantic chunking creates ~30-40 chunks
        # We select top 7 per goal × 5 goals = 35 chunks, ~25 unique after dedup
        # Average chunk size: ~500 chars, so ~25 × 500 = 12,500 chars
        # Plus prompt overhead: ~1,500 chars
        nlp_content_estimate = 14000  # Conservative estimate
        nlp_tokens = estimate_tokens(nlp_content_estimate)
        paper_info['nlp_input_tokens'] = nlp_tokens
        
        # Embedding tokens: all chunks + 5 goal queries (queries only computed once, but counted per paper for fair comparison)
        # Estimate: ~40 chunks × 500 chars = 20,000 chars = 5,000 tokens
        nlp_embed_tokens = estimate_tokens(len(content)) + (5 * 100)  # Full paper chunked + 5 queries
        paper_info['nlp_embed_tokens'] = nlp_embed_tokens
        
        results['nlp']['papers'].append(paper_info.copy())
        results['nlp']['total_input_tokens'] += nlp_tokens
        results['nlp']['total_embed_tokens'] += nlp_embed_tokens
    
    num_papers = len(paper_paths)
    
    # Calculate costs
    print(f"\nAnalyzed {num_papers} papers")
    print("=" * 80)
    print("\nAPPROACH COMPARISON:\n")
    
    # Approach 1: extract_ai.py
    print("1️⃣  extract_ai.py (Direct LLM)")
    print("-" * 80)
    ai_total_input = results['ai']['total_input_tokens']
    ai_total_output = num_papers * AVG_OUTPUT_TOKENS
    ai_input_cost = (ai_total_input / 1_000_000) * INPUT_COST_PER_1M
    ai_output_cost = (ai_total_output / 1_000_000) * OUTPUT_COST_PER_1M
    ai_total_cost = ai_input_cost + ai_output_cost
    
    print(f"   Input tokens:     {ai_total_input:,} ({ai_total_input/num_papers:,.0f} per paper)")
    print(f"   Output tokens:    {ai_total_output:,} ({AVG_OUTPUT_TOKENS:,} per paper)")
    print(f"   Input cost:       ${ai_input_cost:.2f}")
    print(f"   Output cost:      ${ai_output_cost:.2f}")
    print(f"   TOTAL COST:       ${ai_total_cost:.2f}")
    print(f"   Cost per paper:   ${ai_total_cost/num_papers:.2f}")
    
    # Approach 2: extract_nlp.py
    print("\n2️⃣  extract_nlp.py (NLP Semantic Chunking)")
    print("-" * 80)
    nlp_total_input = results['nlp']['total_input_tokens']
    nlp_total_output = num_papers * AVG_OUTPUT_TOKENS
    nlp_total_embed = results['nlp']['total_embed_tokens']
    nlp_input_cost = (nlp_total_input / 1_000_000) * INPUT_COST_PER_1M
    nlp_output_cost = (nlp_total_output / 1_000_000) * OUTPUT_COST_PER_1M
    nlp_embed_cost = (nlp_total_embed / 1_000_000) * EMBED_COST_PER_1M
    nlp_total_cost = nlp_input_cost + nlp_output_cost + nlp_embed_cost
    
    print(f"   Embedding tokens: {nlp_total_embed:,} ({nlp_total_embed/num_papers:,.0f} per paper)")
    print(f"   Input tokens:     {nlp_total_input:,} ({nlp_total_input/num_papers:,.0f} per paper)")
    print(f"   Output tokens:    {nlp_total_output:,} ({AVG_OUTPUT_TOKENS:,} per paper)")
    print(f"   Embedding cost:   ${nlp_embed_cost:.2f}")
    print(f"   Input cost:       ${nlp_input_cost:.2f}")
    print(f"   Output cost:      ${nlp_output_cost:.2f}")
    print(f"   TOTAL COST:       ${nlp_total_cost:.2f}")
    print(f"   Cost per paper:   ${nlp_total_cost/num_papers:.2f}")
    
    # Comparison
    print("\n📊 COMPARISON:")
    print("=" * 80)
    token_reduction = ((ai_total_input - nlp_total_input) / ai_total_input) * 100
    cost_savings = ((ai_total_cost - nlp_total_cost) / ai_total_cost) * 100
    cost_diff = ai_total_cost - nlp_total_cost
    
    print(f"   Token reduction:  {token_reduction:.1f}% fewer input tokens with NLP approach")
    print(f"   Cost savings:     {cost_savings:.1f}% cheaper with NLP approach")
    print(f"   Total savings:    ${cost_diff:.2f} for {num_papers} papers")
    print(f"   Papers per $10:   {10/ai_total_cost*num_papers:.0f} (AI) vs {10/nlp_total_cost*num_papers:.0f} (NLP)")
    
    if nlp_total_cost < ai_total_cost:
        print(f"\n   ✅ NLP approach saves ${cost_diff:.2f} ({cost_savings:.1f}%)")
        print(f"   For 100 papers: Save ~${cost_diff/num_papers*100:.2f}")
        print(f"   For 1000 papers: Save ~${cost_diff/num_papers*1000:.2f}")
    else:
        print(f"\n   Note: Direct LLM approach is cheaper for these papers")
    
    print("\n" + "=" * 80)
    
    # Show per-paper breakdown for first few papers
    print("\nPER-PAPER BREAKDOWN (first 5 papers):")
    print("=" * 80)
    print(f"{'Paper Name':<45} {'Size':<10} {'AI Tokens':<12} {'NLP Tokens':<12} {'Savings':<10}")
    print("-" * 80)
    
    for i in range(min(5, num_papers)):
        paper = results['ai']['papers'][i]
        ai_tokens = paper['ai_input_tokens']
        nlp_tokens = results['nlp']['papers'][i]['nlp_input_tokens']
        savings = ((ai_tokens - nlp_tokens) / ai_tokens * 100) if ai_tokens > 0 else 0
        
        name = paper['name'][:43]
        size = f"{paper['chars']:,}"
        print(f"{name:<45} {size:<10} {ai_tokens:<12,} {nlp_tokens:<12,} {savings:>6.1f}%")
    
    if num_papers > 5:
        print(f"... and {num_papers - 5} more papers")
    
    print("\n" + "=" * 80)
    print("\n💡 RECOMMENDATION:")
    if cost_savings > 30:
        print("   ⭐ Strongly recommend using extract_nlp.py (NLP approach)")
        print("   You'll save significant costs while maintaining quality.")
    elif cost_savings > 10:
        print("   ✓ Recommend using extract_nlp.py (NLP approach)")
        print("   Good cost savings with similar quality.")
    else:
        print("   Either approach is fine for your corpus size.")
    
    print("\n" + "=" * 80)


def main():
    """Main function to run the comparison."""
    base_dir = Path(__file__).parent
    parsed_files_dir = base_dir / 'parsedFilesLlama'
    
    text_files = sorted(parsed_files_dir.glob('*.txt'))
    
    if not text_files:
        print("ERROR: No text files found in parsedFilesLlama directory!")
        return
    
    print("=" * 80)
    print("APPROACH COMPARISON: extract_ai.py vs extract_nlp.py")
    print("=" * 80)
    print("\nThis script estimates token usage and costs for both approaches.")
    print("Note: These are estimates. Actual costs may vary.\n")
    
    estimate_approach_costs(text_files)


if __name__ == "__main__":
    main()

