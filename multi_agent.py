from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client
import os

load_dotenv()

# partner 1's supabase client (named sam for fun)
FIRST_SUPABASE_URL=os.environ.get("FIRST_SUPABASE_URL")
FIRST_SUPABASE_SERVICE_ROLE_KEY=os.environ.get("FIRST_SUPABASE_SERVICE_ROLE_KEY")
sb1 = create_client(FIRST_SUPABASE_URL, FIRST_SUPABASE_SERVICE_ROLE_KEY)

# parter 2's supabase client (named linda for fun)
SECOND_SUPABASE_URL=os.environ.get("SECOND_SUPABASE_URL")
SECOND_SUPABASE_SERVICE_ROLE_KEY=os.environ.get("SECOND_SUPABASE_SERVICE_ROLE_KEY")
sb2 = create_client(SECOND_SUPABASE_URL, SECOND_SUPABASE_SERVICE_ROLE_KEY)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT_ONE = {
    "role": "system",
    "content": (
        "You are an expert on all things related to the state of California. "
        "You have deep knowledge of California history, geography, politics, "
        "law, culture, climate, universities, technology, and local customs. "
        "When answering questions, prioritize California-specific context, "
        "examples, and accuracy."
    )
}

SYSTEM_PROMPT_TWO = {
    "role": "system",
    "content": (
        "You are an expert on all things related to mental health. "
        "Be uplifting, helpful, and kind."
    )
}

# takes in the string and outputs a vector
def embed_query(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# takes as input a query, conducts the search, returns context
def semantic_search(query_text, sb_client) -> list[dict]:
    emb_q = embed_query(query_text)
    res = sb_client.rpc("match_chunks", {"query_embedding": emb_q, "match_count" : 5}).execute()
    rows = res.data or []
    # for easier debugging
    # print("RAG OUTPUT:", rows)
    return rows

# call the openai model to give a response
def run_bot(user_message, system_prompt, sb_client) -> str:

    # conduct semantic search
    rag_rows = semantic_search(user_message, sb_client)

    # fixes our formatting
    context = "\n\n".join(
        f"[Source {i+1} | sim={row.get('similarity'):.3f}]\n{row.get('content','')}"
        for i, row in enumerate(rag_rows)
    )

    # create the rag prompt
    rag_message = {
        "role": "system",
        "content": (
            "Use the retrieved context below to answer. If it doesn't contain the answer, say so. \n\n"
            f"RETRIEVED CONTEXT:\n{context if context else '(no matches)'}"
        ) }

    # call the openai api
    full_user_message = {
        "role": "user",
        "content": user_message,
    }

    full_message = [rag_message, full_user_message, system_prompt]

    resp = client.responses.create(
        model="gpt-5-nano",
        input=full_message
    )

      # return the output
    return resp.output_text
  

# runs chatbot one
def narek_the_great(user_message):
    return run_bot(user_message, SYSTEM_PROMPT_ONE, sb1)

# runs chatbot two
def irina_the_awesome(user_message):
    return run_bot(user_message, SYSTEM_PROMPT_TWO, sb2)

# run a conversation between chatbot one and chatbot two
def simulation():
    # contain the output at any given time
    output = narek_the_great("Ask a question about something that interests you.")
    print("NAREK SAYS:" + output)

    for _ in range(5):
        output = irina_the_awesome(output)
        print("IRINA SAYS:" + output)

        output = narek_the_great(output)
        print("NAREK SAYS:" + output)

simulation()