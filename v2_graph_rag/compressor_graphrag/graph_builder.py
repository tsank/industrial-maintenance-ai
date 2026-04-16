import sys
import json
import re
from pathlib import Path

import spacy
import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(Path(__file__).parent.parent / '.env', override=True)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'maintenance-rag' / 'src'))
from pdf_loader import load_pdf_robust, load_pdf_semantic

nlp = spacy.load('en_core_web_sm')

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

GRAPH_PATH = Path(__file__).parent.parent / 'data' / 'graph.json'
PDF_PATH = Path(__file__).parent.parent.parent / 'maintenance-rag' / 'docs' / '517ba4.pdf'


# ── 1. spaCy extraction ──────────────────────────────────────────────────────

def extract_entities_spacy(text: str) -> list[tuple[str, str]]:
    """Return (entity_text, entity_label) pairs from spaCy NER."""
    doc = nlp(text)
    return [(ent.text.strip(), ent.label_) for ent in doc.ents
            if len(ent.text.strip()) > 2]


# ── 2. LLM extraction ────────────────────────────────────────────────────────

TRIPLE_PROMPT = """You are an expert in industrial compressor systems.
Extract knowledge graph triples from the text below.
Return ONLY a JSON array of triples, each as [subject, relation, object].
Focus on: components, properties, actions, failures, and maintenance procedures.
Return at most 10 triples. If nothing useful, return [].

Text:
{text}

JSON array of triples:"""


def extract_triples_llm(text: str) -> list[tuple[str, str, str]]:
    """Use GPT-4o-mini to extract (subject, relation, object) triples."""
    try:
        response = llm.invoke(TRIPLE_PROMPT.format(text=text[:1500]))
        raw = response.content.strip()
        raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
        triples = json.loads(raw)
        return [(t[0], t[1], t[2]) for t in triples
                if len(t) == 3 and all(isinstance(x, str) for x in t)]
    except Exception as e:
        print(f"  LLM extraction failed: {e}")
        return []


# ── 3. Graph builder ─────────────────────────────────────────────────────────

def build_graph(pdf_path: Path, max_chunks: int = None) -> nx.DiGraph:
    """
    Load PDF chunks, run both extractors, build a NetworkX directed graph.
    Each node has a 'label' and 'type' attribute.
    Each edge has a 'relation' attribute.
    """
    print(f"Loading PDF: {pdf_path}")
    docs = load_pdf_semantic(str(pdf_path), pdf_path.name)
    chunks = docs[:max_chunks] if max_chunks else docs
    print(f"Processing {len(chunks)} chunks...")

    G = nx.DiGraph()

    for i, doc in enumerate(chunks):
        text = doc.page_content
        print(f"  Chunk {i+1}/{len(chunks)}", end='\r')

        # spaCy: add entities as nodes
        entities = extract_entities_spacy(text)
        for ent_text, ent_label in entities:
            ent_text = ent_text.replace('\n', ' ').strip()
            if not G.has_node(ent_text):
                G.add_node(ent_text, label=ent_text, type=ent_label)

        # LLM: add triples as edges
        triples = extract_triples_llm(text)
        for subj, rel, obj in triples:
            if not G.has_node(subj):
                G.add_node(subj, label=subj, type='CONCEPT')
            if not G.has_node(obj):
                G.add_node(obj, label=obj, type='CONCEPT')
            G.add_edge(subj, obj, relation=rel)

    print(f"\nGraph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# ── 4. Persist / load ────────────────────────────────────────────────────────

def save_graph(G: nx.DiGraph, path: Path = GRAPH_PATH):
    data = nx.node_link_data(G, edges="edges")
    path.write_text(json.dumps(data, indent=2))
    print(f"Graph saved to {path}")


def load_graph(path: Path = GRAPH_PATH) -> nx.DiGraph:
    data = json.loads(path.read_text())
    return nx.node_link_graph(data,edges="edges")


# ── 5. Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    G = build_graph(PDF_PATH, max_chunks=None)   # start with 10 to test
    save_graph(G)
    print(f"Sample nodes: {list(G.nodes)[:10]}")
    print(f"Sample edges: {list(G.edges(data=True))[:5]}")