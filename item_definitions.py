import os
import sys
from pathlib import Path
import csv
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# load env vars
load_dotenv()

# init openai client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLUMNS_TO_DEFINE = [
    "AI Features",
    "Performance Degradation Types",
    "Causal Links",
]


def normalize_term(term: str) -> str:
    return " ".join(term.strip().split())


def get_definition(term: str, column: str) -> str:
    """Return a short, plain-English definition for a term."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You write short, plain-English definitions. "
                    "Return a single concise sentence fragment (8-15 words). "
                    "Do not use quotes or parentheses."
                ),
            },
            {
                "role": "user",
                "content": f"Define this term from column '{column}': {term}",
            },
        ],
        temperature=0.2,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


def collect_unique_terms(rows: List[Dict[str, str]]) -> Dict[str, List[str]]:
    terms_by_column: Dict[str, List[str]] = {col: [] for col in COLUMNS_TO_DEFINE}
    seen: Dict[str, set] = {col: set() for col in COLUMNS_TO_DEFINE}

    for row in rows:
        for column in COLUMNS_TO_DEFINE:
            if column not in row:
                continue
            raw_value = row.get(column, "") or ""
            term = normalize_term(raw_value)
            if not term:
                continue
            if term not in seen[column]:
                seen[column].add(term)
                terms_by_column[column].append(term)

    return terms_by_column


def build_definitions(terms_by_column: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
    definition_cache: Dict[str, str] = {}
    definitions: Dict[str, Dict[str, str]] = {col: {} for col in COLUMNS_TO_DEFINE}

    for column, terms in terms_by_column.items():
        for term in terms:
            cache_key = f"{column}||{term}"
            if cache_key not in definition_cache:
                definition_cache[cache_key] = get_definition(term, column)
            definitions[column][term] = definition_cache[cache_key]

    return definitions


def write_definitions_doc(output_path: Path, definitions: Dict[str, Dict[str, str]]) -> None:
    lines = []
    lines.append("# Term Definitions")
    lines.append("")

    for column in COLUMNS_TO_DEFINE:
        lines.append(f"## {column}")
        lines.append("")
        for term, definition in definitions.get(column, {}).items():
            lines.append(f"- {term}")
            lines.append(f"  - {definition}")
            lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_definitions_doc(input_csv: Path, output_doc: Path) -> None:
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not fieldnames:
        raise ValueError("CSV has no header row.")

    terms_by_column = collect_unique_terms(rows)
    definitions = build_definitions(terms_by_column)
    write_definitions_doc(output_doc, definitions)


def main() -> None:
    base_dir = Path(__file__).parent
    input_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else base_dir / "ai_degradation_evidence_new.csv"
    output_doc = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else input_csv.with_name(f"{input_csv.stem}_definitions.md")
    )

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Reading: {input_csv}")
    print(f"Writing: {output_doc}")
    generate_definitions_doc(input_csv, output_doc)
    print("Done.")


if __name__ == "__main__":
    main()
