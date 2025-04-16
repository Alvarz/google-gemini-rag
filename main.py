#!/usr/bin/env python3

import os
from openai import OpenAI
from embeddings import retrieve_documents 
from cost_estimation import init_log, log_usage, count_tokens

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = os.getenv('GEMINI_API_URL')
MODEL = os.getenv('MODEL')

client = OpenAI(
            api_key=GEMINI_API_KEY,  # Google Gemini API key
            base_url=GEMINI_API_URL,  # Gemini base URL
)


def generate_answer_streaming(query, temperature=0.7):
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    messages=[
        {"role": "system", "content": f"Answer the question based on this context:\n{context}"},
        {"role": "user", "content": query}
    ]

    # Pre-calculate input tokens
    system_tokens = count_tokens(messages[0]['content'], 'gpt-4o')
    user_tokens = count_tokens(messages[1]['content'], 'gpt-4o')
    input_tokens = system_tokens + user_tokens
    print(f"Pre-call estimate: input: {input_tokens} tokens - user {user_tokens} tokens - system {system_tokens} tokens")

    # Generate response
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
         max_tokens=1000
    )

    collected_chunks = []
    completion_tokens = 0
    completion_input_tokens = input_tokens
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            collected_chunks.append(chunk.choices[0].delta.content)
        
        # Track tokens from the last chunk (contains usage data)
        if chunk.usage:
            completion_input_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens
        else:
            # workaround because we are not alway (never?) getting the usage data on streaming
            completion_tokens = count_tokens(''.join(collected_chunks), 'gpt-4o')

    
    log_usage(
        model=MODEL,
        endpoint="chat/completions",
        pre_input_tokens=input_tokens,
        system_tokens=system_tokens,
        user_tokens=user_tokens,
        input_tokens=completion_input_tokens,
        output_tokens=completion_tokens
    )


    return ''.join(collected_chunks)

def generate_answer(query, temperature=0.7):
    # Retrieve relevant documents
    relevant_docs = retrieve_documents(query)
    context = "\n".join(relevant_docs)
    messages=[
        {"role": "system", "content": f"Answer the question based on this context:\n{context}"},
        {"role": "user", "content": query}
    ]
    
    # Pre-call token estimation
    system_tokens = count_tokens(messages[0]['content'], 'gpt-4o')
    user_tokens = count_tokens(messages[1]['content'], 'gpt-4o')
    input_tokens = system_tokens + user_tokens
    print(f"Pre-call estimate: input: {input_tokens} tokens - user {user_tokens} tokens - system {system_tokens} tokens")

    
    # Generate response
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
         max_tokens=1000
    )

    log_usage(
        model=MODEL,
        endpoint="chat/completions",
        pre_input_tokens=input_tokens,
        system_tokens=system_tokens,
        user_tokens=user_tokens,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens
    )


    return response.choices[0].message.content



# Example usage:
if __name__ == "__main__":
        init_log()
        queries = [
                # "What is the capital of France?",
                # "What is AI?",
                "What does stoicism says about virtue?"
        ]

        for query in queries:
                print(f"\nQuestion: {query}")
                answer = generate_answer(query)
                # answer = generate_answer_streaming(query)
                print(f"Answer: {answer}")

