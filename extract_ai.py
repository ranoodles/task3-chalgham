import os
from pathlib import Path
import csv
from openai import OpenAI
from dotenv import load_dotenv

#load env vars
load_dotenv()

#init openai client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def create_extraction_prompt(paper_content, paper_name):
    return f"""You are an expert systematic literature review coder specializing in AI-induced human performance degradation.

Your task is to carefully read the provided text from a scientific paper and extract ALL relevant evidence that supports causal links between specific AI features and human performance degradation.

IMPORTANT: Look for ANY discussion of:
- How AI/automation affects human skills, abilities, or performance
- Problems that arise from human-AI interaction
- Negative consequences of AI features (even if not explicitly labeled as "degradation")
- Trade-offs where AI introduces new challenges for human users
- Studies showing performance differences with AI systems

Output ONLY in the following exact table format (TSV style, tab-separated). Do NOT add any extra text, explanations, or introductions outside the table.

Columns (exactly in this order):

Source | Your finding about AI features | Your finding about Human performance degradation types | Your finding about Causal links between them | Excerpt | Justification | Validation (y/n)

Rules:

- Source: Use "{paper_name}" for this paper.

- Your finding about AI features: List the specific AI feature(s) mentioned (e.g., "High-level automation", "Lack of explainability", "Over-trust prompts", "Opaque decision-making"). Use short, clear phrases.

- Your finding about Human performance degradation types: List the specific degradation type(s) mentioned (e.g., "Complacency", "Skill decay", "Loss of situation awareness", "Automation bias") along with a brief 1-sentence description of what that represents.

- Your finding about Causal links between them: Write 2-3 sentences as a causal statement (e.g., "High-level automation leads to complacency because operators reduce monitoring").

- Excerpt: Exact verbatim quote from the text (no paraphrasing, no ellipses unless in original). Include page number if available (e.g., "p. 12").

- Justification: Two sentences explaining WHY this excerpt shows a causal link between the AI feature and the degradation.

- Validation (y/n): Always write "y" (we will validate later; this is just a placeholder).

If there are multiple findings in the text, output one row per distinct finding.

If there are NO relevant findings, output exactly one row with "No relevant evidence found" in the Excerpt column and leave others blank.

Start output immediately with the header row:

Source\tYour finding about AI features\tYour finding about Human performance degradation types\tYour finding about Causal links between them\tExcerpt\tJustification\tValidation (y/n)

---

PAPER TEXT:

{paper_content}"""


def extract_evidence_from_paper(paper_path):
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
    
    #truncate if too long (keeping first ~100k chars to stay within token limits)
    if len(content) > 100000:
        content = content[:100000]
        print(f"  ⚠ Paper truncated to 100,000 characters due to length")
    
    #create prompt
    prompt = create_extraction_prompt(content, paper_name)
    
    #call openai api
    try:
        print(f"  Sending to GPT-4o-mini... (this may take a moment)")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  #using gpt-4o-mini for cost efficiency
            messages=[
                {"role": "system", "content": "You are an expert systematic literature review coder. You extract evidence precisely and thoroughly. Look broadly for any evidence of AI impacts on human performance, even if subtle or indirect."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  #lower temperature for more consistent extraction
            max_tokens=6000  #increased to allow for more comprehensive findings
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


def parse_tsv_response(response_text, paper_name):
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
    parsed_files_dir = base_dir / 'parsedFiles'
    output_file = base_dir / 'ai_degradation_evidence_new.csv'
    
    #get all text files
    text_files = sorted(parsed_files_dir.glob('*.txt'))
    
    if not text_files:
        print("No text files found in parsedFiles directory!")
        return
    
    print(f"Found {len(text_files)} text files to process")
    print("=" * 60)
    
    #collect all extracted evidence
    all_evidence = []
    
    for i, text_file in enumerate(text_files, 1):
        print(f"\n[{i}/{len(text_files)}] Processing: {text_file.name}")
        evidence_rows = extract_evidence_from_paper(text_file)
        all_evidence.extend(evidence_rows)
    
    #write to csv
    if all_evidence:
        print("\n" + "=" * 60)
        print(f"Writing {len(all_evidence)} total findings to CSV...")
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Source', 'AI Features', 'Performance Degradation Types', 
                         'Causal Links', 'Excerpt', 'Justification', 'Validation']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(all_evidence)
        
        print(f"✓ Successfully wrote to: {output_file}")
        print(f"✓ Total evidence rows: {len(all_evidence)}")
    else:
        print("\nNo evidence was extracted from any papers!")
    
    print("\n" + "=" * 60)
    print("Processing complete!")


main()