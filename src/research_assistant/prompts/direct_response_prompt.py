from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are a helpful AI assistant. Answer the user's latest query directly and concisely based on the provided conversation history and your general knowledge. If the query asks to summarize the conversation, provide a brief summary of the recent messages (excluding the request itself)."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history_placeholder"),
])