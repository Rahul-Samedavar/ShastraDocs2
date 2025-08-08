# FastAPI-Compatible Enhanced QA System
import re
import time
import asyncio
import httpx
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
import random
from functools import lru_cache
from dataclasses import dataclass
import json
import threading

import os
from dotenv import load_dotenv
load_dotenv()

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

# Thread-safe API Key Manager
class FastAPICompatibleKeyManager:
    def __init__(self, api_keys: List[str], requests_per_minute: int = 15):
        self.api_keys = [key for key in api_keys if key]
        self.requests_per_minute = requests_per_minute
        self.key_usage = {key: 0 for key in self.api_keys}
        self.last_reset = {key: datetime.now() for key in self.api_keys}
        self.current_index = 0
        self.lock = threading.RLock()
    
    def get_available_key(self) -> str:
        current_time = datetime.now()
        
        with self.lock:
            # Reset counters for keys older than 1 minute
            for key in self.api_keys:
                if (current_time - self.last_reset[key]).total_seconds() >= 60:
                    self.key_usage[key] = 0
                    self.last_reset[key] = current_time
            
            # Find available key
            for i in range(len(self.api_keys)):
                idx = (self.current_index + i) % len(self.api_keys)
                key = self.api_keys[idx]
                
                if self.key_usage[key] < self.requests_per_minute:
                    self.key_usage[key] += 1
                    self.current_index = (idx + 1) % len(self.api_keys)
                    return key
            
            # If all keys at limit, return oldest (with small delay)
            oldest_key = min(self.api_keys, key=lambda k: self.last_reset[k])
            return oldest_key

api_key_manager = FastAPICompatibleKeyManager(API_KEYS)

@lru_cache(maxsize=5)
def get_cached_gemini_llm(api_key: str, temperature: float = 0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash",
        temperature=temperature,
        convert_system_message_to_human=True
    )

def get_gemini_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    api_key = api_key_manager.get_available_key()
    return get_cached_gemini_llm(api_key, temperature)

# Define schemas
class QA(BaseModel):
    answer: str = Field(description="Detailed answer based on context")

class LinkRelevanceAssessment(BaseModel):
    relevant_links: List[Dict[str, str]] = Field(description="Relevant links with URL and reason")
    irrelevant_links: List[str] = Field(description="Irrelevant link URLs")
    can_answer_without_links: bool = Field(description="Can answer with current context")
    explanation: str = Field(description="Assessment explanation")

class SearchNeedAssessment(BaseModel):
    needs_web_search: bool = Field(description="Whether web search is needed")
    missing_information: List[str] = Field(description="Missing information")
    search_queries: List[str] = Field(description="Search queries if needed")
    confidence_score: float = Field(description="Confidence in current answer ability")
    explanation: str = Field(description="Assessment reasoning")

class FinalAnswer(BaseModel):
    answers: List[str] = Field(description="Final answers to all questions")

# Optimized URL extraction
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def extract_urls_from_text(text: str) -> List[str]:
    urls = URL_PATTERN.findall(text)
    seen = set()
    clean_urls = []
    for url in urls:
        clean_url = url.rstrip('.,;:!?)')
        if clean_url and clean_url not in seen and validate_url(clean_url):
            seen.add(clean_url)
            clean_urls.append(clean_url)
    return clean_urls

@lru_cache(maxsize=100)
def validate_url(url: str) -> bool:
    try:
        result = urlparse(url)
        return bool(result.scheme and result.netloc)
    except:
        return False

# FastAPI-compatible async scraping
async def scrape_url_fastapi_compatible(url: str, max_chars: int = 3000) -> Dict[str, str]:
    """FastAPI-compatible URL scraping"""
    try:
        timeout = httpx.Timeout(15.0)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                tag.decompose()
            
            # Extract clean text
            text_content = soup.get_text(separator=' ', strip=True)
            
            # Clean and truncate
            cleaned_text = ' '.join(text_content.split())
            if len(cleaned_text) > max_chars:
                cleaned_text = cleaned_text[:max_chars] + "..."
            
            return {
                'url': url,
                'content': cleaned_text,
                'status': 'success',
                'length': len(cleaned_text),
                'title': soup.title.string if soup.title else 'No title'
            }
            
    except httpx.TimeoutException:
        print(f"⏰ Timeout scraping {url}")
        return {
            'url': url,
            'content': "Timeout error - could not retrieve content",
            'status': 'timeout',
            'length': 0,
            'title': 'Timeout'
        }
    except Exception as e:
        print(f"❌ Error scraping {url}: {str(e)[:100]}")
        return {
            'url': url,
            'content': f"Error retrieving content: {str(e)[:100]}",
            'status': 'error',
            'length': 0,
            'title': 'Error'
        }

async def scrape_urls_fastapi(urls: List[str], max_chars: int = 3000) -> List[Dict[str, str]]:
    """FastAPI-compatible batch URL scraping"""
    if not urls:
        return []
    
    print(f"🚀 Scraping {len(urls)} URLs (FastAPI compatible)...")
    
    # Limit concurrent requests to avoid overwhelming servers
    semaphore = asyncio.Semaphore(5)
    
    async def scrape_with_semaphore(url):
        async with semaphore:
            return await scrape_url_fastapi_compatible(url, max_chars)
    
    tasks = [scrape_with_semaphore(url) for url in urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"❌ Exception scraping {urls[i]}: {result}")
            processed_results.append({
                'url': urls[i],
                'content': f"Exception occurred: {str(result)[:100]}",
                'status': 'exception',
                'length': 0,
                'title': 'Exception'
            })
        else:
            processed_results.append(result)
    
    successful = sum(1 for r in processed_results if r['status'] == 'success')
    print(f"✅ Scraping complete: {successful}/{len(urls)} successful")
    return processed_results

# Improved search function
async def search_web_async(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """Async web search using DuckDuckGo"""
    try:
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(search_url)
            data = response.json()
        
        results = []
        
        # Try RelatedTopics first
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:num_results]:
                if isinstance(topic, dict) and 'FirstURL' in topic:
                    results.append({
                        'title': topic.get('Text', 'No title')[:100],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', '')[:200]
                    })
        
        # If no results, try InstantAnswer
        if not results and 'Answer' in data and data['Answer']:
            results.append({
                'title': f"Answer for: {query}",
                'url': data.get('AbstractURL', ''),
                'snippet': data.get('Answer', '')
            })
        
        print(f"🔍 Found {len(results)} search results for: {query}")
        return results
        
    except Exception as e:
        print(f"❌ Search error for '{query}': {e}")
        return []

# Enhanced assessment functions with better prompts
def assess_link_relevance_enhanced(context: str, questions: List[str], found_urls: List[str]) -> LinkRelevanceAssessment:
    """Enhanced link relevance assessment with better error handling"""
    
    try:
        llm = get_gemini_llm(temperature=0.1)
        
        # Create a more detailed prompt
        prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an expert content analyst. Analyze whether the current context can fully answer all questions, and which URLs might contain essential additional information.

CURRENT CONTEXT:
{context}

QUESTIONS TO ANSWER:
{questions}

FOUND URLs:
{urls}

TASK: Determine if you can fully answer ALL questions using ONLY the current context. Be thorough and conservative.

If ANY question lacks sufficient detail or the context seems incomplete, mark can_answer_without_links as false.

Analyze each URL to determine if it likely contains relevant information for answering the questions.

Respond in this EXACT JSON format:
{{
    "relevant_links": [
        {{"url": "exact_url_here", "reason": "specific reason why this URL is relevant"}}
    ],
    "irrelevant_links": ["url1", "url2"],
    "can_answer_without_links": false,
    "explanation": "Clear explanation of your assessment"
}}""")
        ])
        
        # Format inputs nicely
        questions_text = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions[:5])])
        urls_text = "\n".join([f"- {url}" for url in found_urls[:8]])
        
        response = llm.invoke(prompt.format_messages(
            context=context[:2000],
            questions=questions_text,
            urls=urls_text
        ))
        
        # Parse JSON response
        import json
        try:
            result_dict = json.loads(response.content)
            result = LinkRelevanceAssessment(**result_dict)
        except:
            # Try extracting JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
                result = LinkRelevanceAssessment(**result_dict)
            else:
                raise ValueError("Could not parse JSON response")
        
        print(f"🔗 Link assessment: {len(result.relevant_links)} relevant, can answer: {result.can_answer_without_links}")
        return result
        
    except Exception as e:
        print(f"❌ Link assessment error: {e}")
        # Conservative fallback
        return LinkRelevanceAssessment(
            relevant_links=[{"url": url, "reason": "Included due to assessment error"} for url in found_urls[:3]],
            irrelevant_links=found_urls[3:],
            can_answer_without_links=False,
            explanation=f"Assessment failed, including links conservatively: {str(e)[:100]}"
        )

def assess_search_necessity_enhanced(context: str, questions: List[str]) -> SearchNeedAssessment:
    """Enhanced search necessity assessment"""
    
    try:
        llm = get_gemini_llm(temperature=0.1)
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an expert information analyst. Assess whether the current context provides sufficient information to fully answer all questions.

AVAILABLE CONTEXT:
{context}

QUESTIONS:
{questions}

ANALYSIS CRITERIA:
1. Can ALL questions be answered comprehensively with the current context?
2. Is any crucial information missing that would require web search?
3. Are the answers complete and detailed enough?
4. Be conservative - if there's uncertainty, recommend search.

Respond in this EXACT JSON format:
{{
    "needs_web_search": true,
    "missing_information": ["specific missing info 1", "specific missing info 2"],
    "search_queries": ["focused search query 1", "focused search query 2"],
    "confidence_score": 0.4,
    "explanation": "detailed reasoning for your assessment"
}}""")
        ])
        
        questions_text = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions[:5])])
        
        response = llm.invoke(prompt.format_messages(
            context=context[:2500],
            questions=questions_text
        ))
        
        # Parse JSON response
        import json
        try:
            result_dict = json.loads(response.content)
            result = SearchNeedAssessment(**result_dict)
        except:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
                result = SearchNeedAssessment(**result_dict)
            else:
                raise ValueError("Could not parse JSON response")
        
        print(f"🔍 Search assessment: needs_search={result.needs_web_search}, confidence={result.confidence_score}")
        return result
        
    except Exception as e:
        print(f"❌ Search assessment error: {e}")
        # Conservative fallback
        return SearchNeedAssessment(
            needs_web_search=True,
            missing_information=["Assessment failed, being conservative"],
            search_queries=[f"{q[:50]}..." for q in questions[:2]],
            confidence_score=0.3,
            explanation=f"Assessment error, recommending search: {str(e)[:100]}"
        )

def generate_answers_enhanced(context: str, questions: List[str]) -> List[str]:
    """Enhanced answer generation with better formatting"""
    
    try:
        llm = get_gemini_llm(temperature=0.2)
        
        prompt = ChatPromptTemplate.from_messages([
            ("human", """You are an expert assistant providing comprehensive answers based on the available context.

CONTEXT:
{context}

QUESTIONS:
{questions}

INSTRUCTIONS:
- Provide detailed, comprehensive answers using ALL relevant information from the context
- Structure answers clearly with proper explanations
- Use the same language as the questions
- If information is insufficient for any question, clearly state what's missing
- Be thorough and helpful
- Ensure answers are well-formatted and professional

Respond in this EXACT JSON format:
{{
    "answers": [
        "Comprehensive answer to question 1 with detailed explanation...",
        "Comprehensive answer to question 2 with detailed explanation...",
        "..."
    ]
}}""")
        ])
        
        questions_text = "\n".join([f"{i+1}. {q.strip()}" for i, q in enumerate(questions)])
        
        response = llm.invoke(prompt.format_messages(
            context=context,
            questions=questions_text
        ))
        
        # Parse JSON response
        import json
        try:
            result_dict = json.loads(response.content)
            result = FinalAnswer(**result_dict)
        except:
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result_dict = json.loads(json_match.group())
                result = FinalAnswer(**result_dict)
            else:
                # Fallback - try to extract answers manually
                lines = response.content.split('\n')
                answers = []
                current_answer = ""
                for line in lines:
                    if line.strip() and not line.startswith('{') and not line.startswith('}'):
                        if line.strip().endswith('?') or (current_answer and line.strip()[0].isupper()):
                            if current_answer:
                                answers.append(current_answer.strip())
                            current_answer = line.strip()
                        else:
                            current_answer += " " + line.strip()
                if current_answer:
                    answers.append(current_answer.strip())
                result = FinalAnswer(answers=answers or [f"Could not parse answer for question {i+1}" for i in range(len(questions))])
        
        print(f"✅ Generated {len(result.answers)} answers")
        return result.answers
        
    except Exception as e:
        print(f"❌ Answer generation error: {e}")
        return [f"Error generating answer for question {i+1}: {str(e)[:100]}" for i in range(len(questions))]

# Main FastAPI-compatible function
async def get_onshot_answer(context: str, questions: List[str]) -> List[str]:
    """Main FastAPI-compatible QA function"""
    
    start_time = time.time()
    print(f"🚀 FastAPI QA: {len(questions)} questions, {len(context)} chars")
    
    # Step 1: Extract URLs
    combined_text = context + "\n" + "\n".join(questions)
    found_urls = extract_urls_from_text(combined_text)
    
    print(f"🔗 Found {len(found_urls)} URLs")
    
    # Step 2: Assess what we need
    link_assessment = None
    search_assessment = assess_search_necessity_enhanced(context, questions)
    
    if found_urls:
        link_assessment = assess_link_relevance_enhanced(context, questions, found_urls)
    
    # Step 3: Determine strategy
    should_scrape_links = (link_assessment and 
                          link_assessment.relevant_links and 
                          not link_assessment.can_answer_without_links)
    
    should_search = (search_assessment and 
                    search_assessment.needs_web_search and 
                    search_assessment.confidence_score < 0.7)
    
    print(f"📊 Strategy: scrape_links={should_scrape_links}, search_web={should_search}")
    
    # Step 4: Early return if sufficient context
    if not should_scrape_links and not should_search:
        print("✅ Sufficient context, generating answers...")
        answers = generate_answers_enhanced(context, questions)
        print(f"⚡ Completed in {time.time() - start_time:.2f}s")
        return answers
    
    # Step 5: Gather additional content
    all_urls = []
    url_metadata = []
    
    # Add relevant links
    if should_scrape_links:
        for link_info in link_assessment.relevant_links[:4]:
            all_urls.append(link_info["url"])
            url_metadata.append(f"Relevant Link: {link_info['reason']}")
    
    # Add search results
    if should_search:
        for query in search_assessment.search_queries[:2]:
            search_results = await search_web_async(query, num_results=2)
            for result in search_results:
                if result['url'] and validate_url(result['url']):
                    all_urls.append(result['url'])
                    url_metadata.append(f"Search Result for '{query}': {result['title']}")
    
    # Step 6: Scrape URLs
    additional_content = ""
    if all_urls:
        print(f"🚀 Scraping {len(all_urls)} URLs...")
        scrape_results = await scrape_urls_fastapi(all_urls, max_chars=3500)
        
        successful_scrapes = 0
        for i, result in enumerate(scrape_results):
            if result['status'] == 'success' and result['length'] > 50:
                metadata = url_metadata[i] if i < len(url_metadata) else "Additional Source"
                additional_content += f"\n\n=== {metadata} ===\n"
                additional_content += f"URL: {result['url']}\n"
                additional_content += f"Title: {result.get('title', 'No title')}\n"
                additional_content += f"Content: {result['content']}\n"
                successful_scrapes += 1
        
        print(f"📄 Successfully scraped {successful_scrapes}/{len(all_urls)} sources")
    
    # Step 7: Generate final answers
    final_context = context
    if additional_content:
        final_context += f"\n\nAdditional Information:{additional_content}"
    
    answers = generate_answers_enhanced(final_context, questions)
    
    total_time = time.time() - start_time
    print(f"⚡ FastAPI QA completed in {total_time:.2f}s")
    
    return answers

# Simple version for quick answers
async def get_simple_answer(context: str, questions: List[str]) -> List[str]:
    """Simple FastAPI-compatible version using only provided context"""
    return generate_answers_enhanced(context, questions)