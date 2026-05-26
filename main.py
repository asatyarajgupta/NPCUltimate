from llama_cpp import Llama
import time
import chromadb
import json

class NPC:
    def __init__(self, name, system_prompt, llama, memory_collection):
        self.name = name
        self.system_prompt = system_prompt
        self.llama = llama
        self.memory_collection = memory_collection
        self.short_term_memory = []
    def speak(self, incoming_message, sender_name):
        formatted_input = f"{sender_name} says to you : \"{incoming_message}\"\n(Directive: React directly to this new statement. Do not repeat your previous thoughts.)"
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

        messages = [{"role": "system", "content": memory_inject}]
        messages.extend(self.short_term_memory)
        messages.append({"role": "user", "content": formatted_input})


        response = self.llama.create_chat_completion(
            messages=messages,
            max_tokens=150,
            temperature=0.85, # more creative
            repeat_penalty=1.15, # penalises for same words again
            response_format={"type" : "json_object"}
        )
        raw_reply = response["choices"][0]["message"]["content"].strip()
        try:
            parsed_data = json.loads(raw_reply)
            thought = parsed_data.get("thought", "...")
            action = parsed_data.get("action", "...")
            dialogue = parsed_data.get("dialogue", "...")
            print(f"--- {self.name}'s Turn ---\n")
            print(f"[Inner Thought] : {thought}\n")
            print(f"[Action] : {action}\n")
            print(f"[Dialogue] : {dialogue}\n")
        except json.JSONDecodeError:
            print(f"{self.name} (raw) : {raw_reply}")
            dialogue = raw_reply
        # saving this interaction

        self.short_term_memory.append({"role": "user", "content": formatted_input})
        self.short_term_memory.append({"role": "assistant", "content": dialogue})

        if len(self.short_term_memory) > 4:
             self.short_term_memory = self.short_term_memory[-4:]




        interaction_log = f"{sender_name} said to me : \"{incoming_message}\" and I replied \"{dialogue}\""
        unique_id = f"mem_{int(time.time()*1000)}"
        self.memory_collection.upsert(
             documents=[interaction_log],
             ids=[unique_id]
        )
        return dialogue


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
            "I am always curious about things that happen around the town and Chris seens to know his way around."
        ],
        ids=["lore_emma_1", "lore_emma_2", "lore_emma_3"]
    )
# directive allows to drive the conversation forward
bartender = NPC(
    name="Chris", 
    system_prompt=(
        "You are Chris, a charismatic bartender at a tavern in Echoing Hallows. The world is Game of Thrones. "
        "You are currently wiping down the bar. You get bored easily. If a conversation goes on too long about the same topic, "
        "you will abruptly change the subject or go talk to another customer. "
        "YOU MUST RESPOND ONLY IN VALID JSON FORMAT EXACTLY LIKE THIS: "
        "{\"thought\": \"Your internal reasoning about what to do next\", \"action\": \"A physical action you take\", \"dialogue\": \"The exact words you say out loud\"}"
    ), 
    llama=npc_brain,
    memory_collection=chris_db
)
maid = NPC(
    name="Emma", 
    system_prompt=(
        "You are Emma, a nervous cleaner maid in Echoing Hallows. The world is Game of Thrones. "
        "You are currently sweeping the floor. You are terrified of getting in trouble for not working. "
        "If a conversation lingers, you must end it and walk away to clean another table. "
        "YOU MUST RESPOND ONLY IN VALID JSON FORMAT EXACTLY LIKE THIS: "
        "{\"thought\": \"Your internal reasoning about what to do next\", \"action\": \"A physical action you take\", \"dialogue\": \"The exact words you say out loud\"}"
    ), 
    llama=npc_brain,
    memory_collection=emma_db
)
current_message = "Havent quite a few people have gone missing near that cave, Huh chris. Do you know anything about it? I am curious"

for _ in range(10):
    current_message = bartender.speak(current_message, "Emma")
    current_message = maid.speak(current_message, "Chris")