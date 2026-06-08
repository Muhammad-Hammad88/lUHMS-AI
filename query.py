import sys
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="lumhs")

question = sys.argv[1]
q = question.lower()

embedding = model.encode(question).tolist()

# ALWAYS A DICT (fixes IDE + Chroma typing issues)
where_filter = {}

if "admission" in q or "apply" in q:
    where_filter = {"type": "admission"}

results = collection.query(
    query_embeddings=[embedding],
    n_results=5,
    where=where_filter,
    include=["documents", "metadatas", "distances"]
)

docs_raw = results.get("documents") or [[]]
docs = docs_raw[0] if docs_raw else []

if docs:
    print("\n\n".join(docs))
else:
    print("")