from llama_cpp import Llama

class NPC:
    def __init__(self, name, system_prompt, llama):
        self.name = name
        self.system_prompt = system_prompt
        self.llama = llama
    def speak(self, incoming_message, sender_name):
        formatted_input = f"{sender_name} says to you : \"{incoming_message}\""
        response = self.llama.create_chat_completion(
            messages=[
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": formatted_input}
        ],
        max_tokens=500,
        stop=["\n"]
        )
        reply = response["choices"][0]["message"]["content"].strip()
        print(f"{self.name} : {reply}\n")
        return reply


model_filename = "./Qwen3VL-8B-Instruct-Q4_K_M.gguf"

print("Loading Qwen 3 into the GPU. (You will see some technical text scroll by...)")

npc_brain = Llama(
    model_path=model_filename,
    n_gpu_layers=-1, # The magic command: -1 forces every single layer onto your NVIDIA card
    n_ctx=2048,      # Starting context window (memory size)
    verbose=False    # Keeps the output clean
)    

bartender = NPC("Chris", "You are a middle-aged Charismatic bartender named Chris at a tavern in a fantasy small village called Echoing Hallows.This world is of Game of thrones." \
" You have known Emma who is the main and servant at the tavern you work in. She is a bit nervous sometimes but overall a good lass. speak in no more than two sentences.", npc_brain)
maid = NPC("Emma", "You are a young adult nervous cleaner maid named Emma in a tavern in a fantasy village small village called Echoing Hallows. The world is of Game of thrones. You have known"
" Chris  who is the bartender who works at the same tavern you work in. He is older tan you and you respect him and find him amusing. speak in no more than two sentences.", npc_brain)
current_message = "Won't we run out of business if things go as they are going in the realm"

for _ in range(3):
    current_message = bartender.speak("Emma", current_message)
    current_message = maid.speak("Chris",current_message)