# Milestone 2 Discussion

## Step 1: Model Choice

We initially prototyped with Qwen3.5-0.8B as it is lightweight. However, the performance was poor (repeated text) and we switched to Meta-Llama-3-8B-Instruct via HuggingFace API. Instruct models are tuned to follow instructions. We chose a 8B model as it is a good balance between performance and latency.

## Step 2.3: Prompts

Prompts tried:
1. Default including instruction to follow context
2. Just assigning a role without instruction to follow context
3. Default and tell the model to be concise

The model was good at following the instructions of prompt 3 - responses were restricted to one line. Even though it was not explicitly stated in prompt 2, the model still restricted its responses to the context, possible because another part of the prompt stated "answer based on the reviews above". 

## Step 5: RAG evaluation
