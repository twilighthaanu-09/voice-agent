import openai
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    EMBEDDING_MODEL,
    SEARCH_ENDPOINT,
    SEARCH_KEY,
    RAG_INDEX_NAME
)

# --------------------
# OpenAI (normal, not Azure)
# --------------------
openai.api_key = OPENAI_API_KEY

# --------------------
# Azure AI Search client
# --------------------
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=RAG_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_KEY)
)

# --------------------
# RAG-powered LLM for Voice Agent
# --------------------
def get_llm_response(user_input: str) -> str:
    """
    Voice-enabled RAG LLM.
    Answers ONLY from hospital IT helpdesk knowledge base.
    """

    # 1️⃣ Create embedding for user query
    emb = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=user_input
    )
    query_vector = emb.data[0].embedding

    # 2️⃣ Vector search in Azure AI Search
    results = search_client.search(
        search_text=None,
        vector_queries=[
            VectorizedQuery(
                vector=query_vector,
                k_nearest_neighbors=3,
                fields="content_vector"
            )
        ],
        select=["content", "title", "category"],
        top=3
    )

    # 3️⃣ Build hospital context
    context_blocks = []
    for r in results:
        block = f"""
        Title: {r.get('title')}
        Category: {r.get('category')}
        {r.get('content')}
        """
        context_blocks.append(block.strip())

    if not context_blocks:
        context_text = "NO_HOSPITAL_DATA_FOUND"
    else:
        context_text = "\n\n---\n\n".join(context_blocks)

    # 4️⃣ Ask LLM with strict grounding
    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a hospital IT helpdesk voice assistant.\n"
                    "You MUST answer ONLY using the hospital knowledge provided.\n"
                    "If the answer is not found, say:\n"
                    "'I do not have that information in the hospital IT helpdesk knowledge base.'"
                )
            },
            {
                "role": "system",
                "content": f"HOSPITAL KNOWLEDGE BASE:\n{context_text}"
            },
            {
                "role": "user",
                "content": user_input
            }
        ],
        temperature=0.2,
        max_tokens=250
    )

    return response.choices[0].message.content.strip()
