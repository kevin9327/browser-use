# 1. Install Ollama: https://github.com/ollama/ollama
# 2. Run `./bin/start-qwen-local.sh` to pull Qwen3.8-27B quants and start the server
# 3. Or manually: `ollama serve` + `ollama pull qwen3.8:27b-q4_K_M`

from browser_use import Agent, ChatOllama

llm = ChatOllama(
	model='qwen3.8:27b-q4_K_M',
	ollama_options={'num_ctx': 32768},
)

Agent('find the founders of browser-use', llm=llm).run_sync()
