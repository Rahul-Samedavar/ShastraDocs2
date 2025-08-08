import google.generativeai as genai
from typing import List
import time


from dotenv import load_dotenv
import os
load_dotenv()
GEMINI_API_KEY_MULTILINGUAL = os.getenv("GEMINI_API_KEY_MULTILINGUAL")
assert GEMINI_API_KEY_MULTILINGUAL, "GEMINI_API_KEY_MULTILINGUAL"
MODEL_NAME = "models/gemini-2.0-flash"



    # Configure the API key
genai.configure(api_key=GEMINI_API_KEY_MULTILINGUAL)

system_prompt = """You are expert query assistant. You are task is to answer the users queries according to the context. Your answers should be only from the context given. and it should be descriptive
Your answers should be in the same language as the users question.  If the context doesn't contain enough information to answer the question, please indicate that clearly."""

# Initialize the model
try:
    model = genai.GenerativeModel(MODEL_NAME, 
                system_instruction=system_prompt)
except Exception as e:
    raise Exception(f"Failed to initialize model {MODEL_NAME}: {str(e)}")


def answer_multilingual(context: str, questions: List[str]) -> List[str]:
    """
    Takes context and a list of questions as input, and returns answers using Gemini 2.0 model.
    
    Args:
        context (str): The context/background information to base answers on
        questions (List[str]): List of questions to be answered
        api_key (str): Your Google AI API key
        model_name (str): Gemini model name (default: "gemini-2.0-flash-exp")
    
    Returns:
        List[str]: List of answers in the same order as input questions
    
    Raises:
        Exception: If there's an error with the API call or model
    """
    
    
    answers = []
    
    for i, question in enumerate(questions):
        try:
            # Create a prompt that includes context and the specific question
            prompt = f"""
Context: {context}

Question: {question}

Answer:"""
            
            # Generate response
            response = model.generate_content(prompt)
            
            # Extract the answer text
            if response.text:
                answer = response.text.strip()
            else:
                answer = "Unable to generate answer - no response received."
            
            answers.append(answer)
            
            # Add a small delay to avoid rate limiting (adjust as needed)
            if i < len(questions) - 1:  # Don't delay after the last question
                time.sleep(0.1)
                
        except Exception as e:
            # Handle individual question errors gracefully
            error_answer = f"Error generating answer: {str(e)}"
            answers.append(error_answer)
            print(f"Warning: Error processing question {i+1}: {str(e)}")
    
    return answers
