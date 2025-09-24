# tools/final_answer_tool.py
from langchain.tools import tool
from pydantic import BaseModel, Field

class FinalAnswerSchema(BaseModel):
    text: str = Field(..., description="The final answer to return to the user.")

@tool("final_answer_tool", return_direct=True, description="Use to finish the run and return the final answer.")
def final_answer_tool(text: str) -> str:
    # Return the text directly to end the agent run.
    return text
