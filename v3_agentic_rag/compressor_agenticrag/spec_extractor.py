# compressor_agenticrag/spec_extractor.py
import json
import math
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "maintenance-rag" / "src"))
from pdf_loader import load_pdf_robust

MANUAL_PATH = Path(__file__).parent.parent.parent / "maintenance-rag" / "docs" / "517ba4.pdf"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "specs.json"

MAX_CHARS_PER_CHUNK = 320_000  # ~100k tokens, leaves headroom for prompt + response

EXTRACTION_PROMPT = """You are extracting technical specifications from an Atlas Copco GA5 compressor manual.

From the text below, extract every concrete technical specification you can find.
These include: pressures, temperatures, voltages, currents, power ratings, flow rates,
weights, dimensions, capacities, speeds, intervals, and any other numeric technical values.

Return ONLY a valid JSON object. Keys must be snake_case, descriptive, and unique.
Values must be strings preserving the original unit (e.g. "13 bar", "5.5 kW", "2.5 litres").
Do not include vague or non-numeric specifications.
Do not include any explanation, preamble, or markdown fences — just the raw JSON object.

Manual text:
{text}
"""


def make_chunks(text: str) -> list[str]:
    if len(text) <= MAX_CHARS_PER_CHUNK:
        return [text]
    n = math.ceil(len(text) / MAX_CHARS_PER_CHUNK)
    size = len(text) // n
    return [text[i * size:(i + 1) * size] for i in range(n)]


def extract_specs() -> dict:
    print("Loading manual...")
    pages = load_pdf_robust(str(MANUAL_PATH), "517ba4.pdf")
    full_text = "\n\n".join(p.page_content for p in pages)

    char_count = len(full_text)
    token_estimate = char_count // 4
    print(f"Extracted text: {char_count:,} characters (~{token_estimate:,} tokens)")

    chunks = make_chunks(full_text)
    print(f"Splitting into {len(chunks)} chunk(s) for extraction.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    merged: dict = {}

    for i, chunk in enumerate(chunks, 1):
        print(f"Extracting specs from chunk {i}/{len(chunks)}...")
        prompt = EXTRACTION_PROMPT.format(text=chunk)
        response = llm.invoke(prompt)
        raw = response.content.strip()

        # Strip accidental markdown fences if the model disobeys
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        try:
            specs = json.loads(raw)
            merged.update(specs)
            print(f"  Found {len(specs)} specs in chunk {i}.")
        except json.JSONDecodeError as e:
            print(f"  Warning: JSON parse failed for chunk {i}: {e}")
            print(f"  Raw output snippet: {raw[:200]}")

    return merged


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    specs = extract_specs()
    print(f"\nTotal specs extracted: {len(specs)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(specs, f, indent=2)

    print(f"Written to {OUTPUT_PATH}")
    print("\nSample entries:")
    for k, v in list(specs.items())[:8]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()