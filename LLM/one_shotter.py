import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from config.config import OPENAI_API_KEY  # Changed from GROQ_API_KEY
import re

# Define the output schema
class QA(BaseModel):
    answer: str = Field(description="Detailed answer to the question based on the given context")

class QABatch(BaseModel):
    answers: List[QA]

# Initialize the OpenAI LLM
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY, 
    model_name="gpt-4o",  # You can also use "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", etc.
    temperature=0  # For consistent responses
)

# Define the structured output parser
from langchain.output_parsers import PydanticOutputParser
parser = PydanticOutputParser(pydantic_object=QABatch)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert insurance assistant. Use the context to answer each question accurately."),
    ("human", """Context:
{context}

Questions:
{questions}

Provide the output as a JSON list in this format:
{format_instructions}
""")
])

def get_onshot_answer(context: str, questions: List[str], retries: int = 3) -> List[str]:
    print(f"context: {context}")
    format_instructions = """
Respond ONLY with a JSON object like this:
{
  "answers": [
    "<Answer to Question 1>",
    "<Answer to Question 2>",
    ...
  ]
}
Each answer should be descriptive and contains sufficient information form context to answer.
in case any wrong information is given in context, your answer should refer the wrong answer given and explain the correct answer.
Do not include any explanation, formatting, or JSON schema.
If the context or questions are in a different language, ensure that the answer to each question is given in the same language as that specific question.
For example if Questions 1 is in Hindi and Question 2 is in English, Answer 1 should be in Hindi and Answer 2 should be in English.
"""
    
    # Adjust parser to expect just list of strings
    class QABatchSimple(BaseModel):
        answers: List[str]
    
    parser = PydanticOutputParser(pydantic_object=QABatchSimple)
    chain = prompt | llm | parser
    
    last_exception = None
    for attempt in range(1, retries + 1):
        print(f"[Attempt {attempt}]")
        try:
            result = chain.invoke({
                "context": context,
                "questions": questions,
                "format_instructions": format_instructions
            })
            return result.answers
        except Exception as e:
            last_exception = e
            print(f"❌ Attempt {attempt} failed: {e}")
    
    raise RuntimeError(f"Failed to get structured answers after {retries} retries.\nLast error: {last_exception}")

# Alternative implementation using OpenAI's structured outputs (if you want to use the newer approach)
def get_onshot_answer_structured(context: str, questions: List[str], retries: int = 3) -> List[str]:
    """
    Alternative implementation using OpenAI's structured outputs feature
    Available for gpt-4o-mini, gpt-4o-2024-08-06, and later models
    """
    print(f"context: {context}")
    
    # Use OpenAI's structured outputs
    llm_structured = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name="gpt-4o",
        temperature=0
    ).with_structured_output(QABatchSimple)
    
    class QABatchSimple(BaseModel):
        answers: List[str]
    
    prompt_structured = ChatPromptTemplate.from_messages([
        ("system", "You are an expert insurance assistant. Use the context to answer each question accurately."),
        ("human", """Context:
{context}

Questions:
{questions}

Each answer should be descriptive and contains sufficient information from context to answer.
In case any wrong information is given in context, your answer should refer the wrong answer given and explain the correct answer.
If the context or questions are in a different language, ensure that the answer to each question is given in the same language as that specific question.
""")
    ])
    
    chain = prompt_structured | llm_structured
    
    last_exception = None
    for attempt in range(1, retries + 1):
        print(f"[Attempt {attempt}]")
        try:
            result = chain.invoke({
                "context": context,
                "questions": questions
            })
            return result.answers
        except Exception as e:
            last_exception = e
            print(f"❌ Attempt {attempt} failed: {e}")
    
    raise RuntimeError(f"Failed to get structured answers after {retries} retries.\nLast error: {last_exception}")