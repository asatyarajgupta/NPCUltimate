from llama_cpp import Llama

# IMPORTANT: Make sure this string exactly matches the filename you downloaded!
model_filename = "./Qwen3VL-8B-Instruct-Q4_K_M.gguf"

print("Loading Qwen 3 into the GPU. (You will see some technical text scroll by...)")

# This initializes the raw C++ engine
npc_brain = Llama(
    model_path=model_filename,
    n_gpu_layers=-1, # The magic command: -1 forces every single layer onto your NVIDIA card
    n_ctx=2048,      # Starting context window (memory size)
    verbose=False    # Keeps the output clean
)

print("\nModel loaded successfully! Thinking...\n")

# We use the built-in chat format since Qwen 3 is a conversational model
response = npc_brain.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a Charismatic bartender at a tavern in a fantasy village. A traveller just walked in."},
        {"role": "user", "content": "Hello there! Do you have any thing cheap for a humble passerby?"}
    ],
    max_tokens=150
)

# Extract and print the AI's response
print("Bartender:", response["choices"][0]["message"]["content"])