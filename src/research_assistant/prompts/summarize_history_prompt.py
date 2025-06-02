from langchain_core.prompts import ChatPromptTemplate

system_prompt = """You are a summarization assistant. Based ONLY on the provided conversation history, create a concise summary of the main points discussed. Start the summary directly without introductory phrases like "Here is a summary...".

**Conversation History:**
{history}

**Concise Summary:**"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt.format(history=history_str)),
        # No human message needed as history is in system prompt
])