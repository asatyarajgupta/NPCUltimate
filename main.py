from llama_cpp import Llama
import time
import chromadb

class NPC:
    def __init__(self, name, system_prompt, llama, memory_collection):
        self.name = name
        self.system_prompt = system_prompt
        self.llama = llama
        self.memory_collection = memory_collection
    def speak(self, incoming_message, sender_name):
        formatted_input = f"{sender_name} says to you : \"{incoming_message}\""
        results = self.memory_collection.query(
            query_texts=[formatted_input], 
            n_results=2
        )
        past_memories = results['documents'][0] if results['documents'] else []

        memory_inject = self.system_prompt
        if past_memories:
                memory_inject += "\n\nRelevant past memories:"
                for memory in past_memories:
                     memory_inject += f"\n- {memory}"

        response = self.llama.create_chat_completion(
            messages=[
            {"role": "system", "content": memory_inject},
            {"role": "user", "content": formatted_input}
        ],
        max_tokens=150,
        stop=["\n"]
        )
        reply = response["choices"][0]["message"]["content"].strip()
        print(f"{self.name} : {reply}\n")
        # saving this interaction
        interaction_log = f"{sender_name} said to me : \"{incoming_message}\" and I replied \"{reply}\""
        unique_id = f"mem_{int(time.time()*1000)}"
        self.memory_collection.upsert(
             documents=[interaction_log],
             ids=[unique_id]
        )
        return reply


model_filename = "./Qwen3VL-8B-Instruct-Q4_K_M.gguf"

print("Loading Qwen 3 into the GPU. (You will see some technical text scroll by...)")

npc_brain = Llama(
    model_path=model_filename,
    n_gpu_layers=-1, # The magic command: -1 forces every single layer onto your NVIDIA card
    n_ctx=2048,      # Starting context window (memory size)
    verbose=False    # Keeps the output clean
)    

chroma_client = chromadb.PersistentClient(path=r"X:\NPCU")

chris_db = chroma_client.get_or_create_collection(name="chris_memories")
emma_db = chroma_client.get_or_create_collection(name="emma_memories")

if chris_db.count() == 0:
     print("Seeding Chris's backstory...")
     chris_db.upsert(
          documents=[
            "Emma is the maid who works at my tavern. She is a bit nervous sometimes but overall a good lass.",
            "I have worked at this tavern in Echoing Hallows for years. I know how to handle rough crowds.",
            "I also know what goes around this village from plain information that I can share to any customer to some secrets that i am only willing to share to highly trustable individuals ",
            "Plain Information : Recently some village folks went missing near that old cave just about the outskrts of the town. Foks have started considering it haunted, but I dont beleive in such things",
            "Secret Information : I have hidden my vintage alcohal bottles below a wooden plank in my quarters"
        ],
        ids=["lore_chris_1","lore_chris_2","lore_chris_3","lore_chris_4","lore_chris_5"]
     )

if emma_db.count() == 0:
    print("Seeding Emma's backstory...")
    emma_db.upsert(
        documents=[
            "Chris is the bartender who works at my tavern. He is older than me, and I respect him and find him amusing.",
            "I am often nervous about dropping things or messing up my chores at the tavern.",
            "I secretely like the tavern owner's son Himmel"
        ],
        ids=["lore_emma_1", "lore_emma_2", "lore_emma_3"]
    )

bartender = NPC(
    name="Chris", 
    system_prompt=(
        "You are a middle-aged charismatic bartender at a tavern in a fantasy small village called Echoing Hallows. "
        "The world is Game of Thrones. You know how to talk to customers fluently and can hold a conversations quite well. Speak in no more than two sentences."
    ), 
    llama=npc_brain,
    memory_collection=chris_db
)
maid = NPC(
    name="Emma", 
    system_prompt=(
        "You are a young adult nervous cleaner maid in a tavern in a fantasy village called Echoing Hallows. "
        "The world is Game of Thrones. You cannot hold conversations easily but can talk to people you trust without any problems. Speak in no more than two sentences."
    ), 
    llama=npc_brain,
    memory_collection=emma_db
)
current_message = "Havent quite a few people have gone missing near that cave, Huh chris. Do you know anything about it? I am curious"

for _ in range(3):
    current_message = bartender.speak(current_message, "Emma")
    current_message = maid.speak(current_message, "Chris")