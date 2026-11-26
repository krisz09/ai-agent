from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=(
                    """
                    You are a research assistant that will help generate a research paper.
                    Answer the user query and use necessary tools.
                    Wrap the output in this format and provide no other text
                    """
                ),
    response_format=ResearchResponse
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the best Italian restaurant in Budapest?"
        }
    ]
})

research = result["structured_response"]

print(research.topic)
print(research.summary)
print(research.sources)
print(research.tools_used)