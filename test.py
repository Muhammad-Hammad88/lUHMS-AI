import chromadb

# =========================
# LOAD DB
# =========================
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("lumhs")

# =========================
# FETCH SAMPLE DATA
# =========================
print("\n🔍 Fetching sample documents...\n")

data = collection.get(
    limit=20,
    include=["documents", "metadatas"]
)

docs = data.get("documents", [])
metas = data.get("metadatas", [])

# =========================
# PRINT ALL ENTRIES
# =========================
for i, (doc, meta) in enumerate(zip(docs, metas)):
    print("=" * 80)
    print(f"INDEX: {i}")

    print("\n📄 DOCUMENT:")
    print(doc)

    print("\n🏷 METADATA:")
    print(meta)

# =========================
# SEARCH FOR KEYWORD (ADMISSION / 2026)
# =========================
print("\n\n🔎 SEARCHING FOR '2026' OR 'ADMISSION'...\n")

for i, doc in enumerate(docs):
    if "2026" in doc.lower() or "admission" in doc.lower():
        print("=" * 80)
        print(f"MATCH FOUND AT INDEX {i}")
        print(doc)

print("\n✅ DONE")