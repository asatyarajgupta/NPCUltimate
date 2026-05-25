import chromadb
chroma_client = chromadb.PersistentClient(path=r"X:\NPCU")
chris_memory = chroma_client.get_or_create_collection(name="chris_memories")
print("seeding Chris's memory database...\n")

chris_memory.upsert(
    documents=[
        "Emma dropped a tray of mugs last week and cried. I gave her a free ale to calm her down.",
        "The Lord Commander of the Night's Watch visited the tavern a month ago and tipped in gold.",
        "I keep my best bottle of Dornish wine hidden under the floorboards behind the bar."
    ],
    ids=["mem1", "mem2", "mem3"]
)

incoming_message = "Emma says to you: 'I am so sorry, Chris, my hands are shaking today. I almost dropped the mugs again!'"

print(f"Trigger Event: {incoming_message}\n")
print(f"Searching Chris's memory for relevant past events...\n")

results = chris_memory.query(query_texts=[incoming_message], n_results=1)

retrieved_memory = results['documents'][0][0]
print(f"relevant memory found: {retrieved_memory}")