import os
from pathlib import Path
import csv
from openai import OpenAI
from dotenv import load_dotenv

#load env vars
load_dotenv()

#init openai client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def determine_industry_domain(row):
    """Use AI to determine the industry domain/field for a given row."""
    
    source = row.get('Source', '')
    ai_features = row.get('AI Features', '')
    excerpt = row.get('Excerpt', '')
    causal_links = row.get('Causal Links', '')
    justification = row.get('Justification', '')
    
    prompt = f"""Carefully read through the following information from a research paper and identify the specific industry domain or field where this AI feature is applied.

Source: {source}
AI Features: {ai_features}
Excerpt: {excerpt}
Causal Links: {causal_links}
Justification: {justification}

Read through all the content above carefully. Based on the context, terminology, examples, and domain-specific references mentioned, determine the specific industry domain or field.

Provide a specific and understandable domain/field name that accurately reflects the context. Be precise - if it's about medical imaging, say "medical imaging" or "radiology", not just "healthcare". If it's about autonomous vehicles, say "autonomous vehicles" or "automotive", not just "transportation". If it's truly general or spans multiple unrelated domains, use "general".

Respond with ONLY the industry domain/field name, nothing else."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at identifying specific industry domains from research paper content. Carefully analyze the context, terminology, and examples to determine the precise domain. Be specific and accurate - avoid generic categories unless truly appropriate. Respond with only the domain name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=50
        )
        
        domain = response.choices[0].message.content.strip()
        # Clean up any quotes or extra text
        domain = domain.strip('"\'')
        return domain
        
    except Exception as e:
        print(f"  ✗ Error determining domain: {e}")
        return "general"


def main():
    #set up paths
    base_dir = Path(__file__).parent
    csv_file = base_dir / 'ai_degradation_evidence_new.csv'
    
    if not csv_file.exists():
        print(f"Error: {csv_file} not found!")
        return
    
    #read the CSV
    print(f"Reading {csv_file}...")
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        #check if Industry Domain/Field column exists
        if 'Industry Domain/Field' not in fieldnames:
            print("Error: 'Industry Domain/Field' column not found in CSV!")
            print(f"Current columns: {fieldnames}")
            return
        
        rows = list(reader)
    
    print(f"Found {len(rows)} rows to process")
    print("=" * 60)
    
    #process each row
    updated_count = 0
    for i, row in enumerate(rows, 1):
        old_domain = row.get('Industry Domain/Field', '').strip()
        
        print(f"\n[{i}/{len(rows)}] Processing row {i}...")
        if old_domain:
            print(f"  Overriding existing domain: {old_domain}")
        print(f"  Source: {row.get('Source', 'N/A')}")
        print(f"  AI Features: {row.get('AI Features', 'N/A')[:80]}...")
        
        domain = determine_industry_domain(row)
        row['Industry Domain/Field'] = domain
        updated_count += 1
        
        print(f"  ✓ Determined domain: {domain}")
    
    #write back to CSV
    print("\n" + "=" * 60)
    print(f"Writing {updated_count} updated rows back to CSV...")
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Successfully updated {csv_file}")
    print(f"✓ Processed {updated_count} row(s)")
    
    print("\n" + "=" * 60)
    print("Processing complete!")


if __name__ == "__main__":
    main()

