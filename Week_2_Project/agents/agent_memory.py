from typing import Dict, List, Any
from pydantic import BaseModel
from langchain_core.memory import BaseMemory
from langchain_core.chat_history import InMemoryChatMessageHistory  # or another ChatMessageHistory impl
from langchain_core.messages import HumanMessage, AIMessage

class ChatHistoryMemory(BaseMemory, BaseModel):
    """
    Keeps an InMemoryChatMessageHistory and implements the BaseMemory API.
    """
    chat_history: InMemoryChatMessageHistory = InMemoryChatMessageHistory()
    memory_key: str = "chat_history"
    return_messages: bool = True
    input_key: str = "input"
    output_key: str = "output"

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        msgs = self.chat_history.messages
        if self.return_messages:
            return {self.memory_key: msgs}
        # fallback: join into single text
        joined = "\n".join(m.content if isinstance(m.content, str) else str(m.content) for m in msgs)
        return {self.memory_key: joined}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        # adapt to your input/output keys; your agent uses input_key/output_key in build_agent
        user_text = inputs.get(self.input_key) or inputs.get("input")
        ai_text = outputs.get(self.output_key) or outputs.get("output")
        if user_text:
            self.chat_history.add_user_message(user_text)
        if ai_text:
            self.chat_history.add_ai_message(ai_text)

    def clear(self) -> None:
        self.chat_history.clear()
