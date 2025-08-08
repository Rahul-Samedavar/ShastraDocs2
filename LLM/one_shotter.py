# Ultra-Fast Enhanced QA System with Optimized Parallel Processing
import re
import time
import requests
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import threading
from datetime import datetime, timedelta
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial, lru_cache
import aiohttp
import aiodns
from dataclasses import dataclass
import json

import os
from dotenv import load_dotenv
load_dotenv()

API_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3")
]

# Optimized API Key Management with Connection Pooling
class OptimizedAPIKeyManager:
    def __init__(self, api_keys: List[str], requests_per_minute: int = 15):
        """Enhanced API key manager with faster rotation logic"""
        self.api_keys = [key for key in api_keys if key]  # Filter None keys
        self.requests_per_minute = requests_per_minute
        self.key_usage = {key: 0 for key in self.api_keys}
        self.last_reset = {key: datetime.now() for key in self.api_keys}
        self.current_index = 0
        self.lock = threading.RLock()  # Reentrant lock for better performance
    
    def get_available_key(self) -> str:
        """Fast key rotation with minimal locking"""
        current_time = datetime.now()
        
        with self.lock:
            # Fast reset check - only reset counters older than 1 minute
            for key in self.api_keys:
                if (current_time - self.last_reset[key]).total_seconds() >= 60:
                    self.key_usage[key] = 0
                    self.last_reset[key] = current_time
            
            # Quick available key search
            for i in range(len(self.api_keys)):
                idx = (self.current_index + i) % len(self.api_keys)
                key = self.api_keys[idx]
                
                if self.key_usage[key] < self.requests_per_minute:
                    self.key_usage[key] += 1
                    self.current_index = (idx + 1) % len(self.api_keys)
                    return key
            
            # If all keys are at limit, use oldest one and add minimal delay
            oldest_key = min(self.api_keys, key=lambda k: self.last_reset[k])
            time.sleep(0.1)  # Minimal delay instead of full minute wait
            return oldest_key

# Global optimized manager
api_key_manager = OptimizedAPIKeyManager(API_KEYS)

# Cached LLM instances for reuse
@lru_cache(maxsize=5)
def get_cached_gemini_llm(api_key: str, temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Cached LLM instances to avoid repeated initialization"""
    return ChatGoogleGenerativeAI(
        google_api_key=api_key,
        model="gemini-2.0-flash",
        temperature=temperature,
        convert_system_message_to_human=True
    )

def get_gemini_llm(temperature: float = 0) -> ChatGoogleGenerativeAI:
    """Get optimized Gemini LLM instance"""
    api_key = api_key_manager.get_available_key()
    return get_cached_gemini_llm(api_key, temperature)

# Define optimized output schemas
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

# Optimized URL extraction with compiled regex
URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

def extract_urls_from_text(text: str) -> List[str]:
    """Optimized URL extraction with compiled regex"""
    urls = URL_PATTERN.findall(text)
    return list(dict.fromkeys(url.rstrip('.,;:!?)') for url in urls if url))

@lru_cache(maxsize=100)
def validate_url(url: str) -> bool:
    """Cached URL validation"""
    try:
        result = urlparse(url)
        return bool(result.scheme and result.netloc)
    except:
        return False

# Async web scraping for maximum speed
async def async_scrape_url(session: aiohttp.ClientSession, url: str, max_chars: int = 2500) -> Dict[str, str]:
    """Async URL scraping for maximum performance"""
    try:
        timeout = aiohttp.ClientTimeout(total=5)  # Aggressive timeout
        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                return {'url': url, 'content': f"HTTP {response.status}", 'status': 'error', 'length': 0}
            
            html = await response.text()
            
            # Fast BeautifulSoup parsing with minimal features
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements efficiently
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            # Get text with aggressive cleanup
            text = ' '.join(soup.stripped_strings)
            
            # Truncate early to save memory
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            return {
                'url': url,
                'content': text,
                'status': 'success',
                'length': len(text)
            }
            
    except Exception as e:
        return {
            'url': url,
            'content': f"Error: {str(e)[:100]}",
            'status': 'error', 
            'length': 0
        }

async def scrape_urls_async(urls: List[str], max_chars: int = 2500) -> List[Dict[str, str]]:
    """Ultra-fast async URL scraping"""
    if not urls:
        return []
    
    print(f"🚀 Async scraping {len(urls)} URLs...")
    
    # Optimized connector with connection pooling
    connector = aiohttp.TCPConnector(
        limit=20,  # Connection pool size
        limit_per_host=5,
        ttl_dns_cache=300,
        use_dns_cache=True
    )
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    async with aiohttp.ClientSession(
        connector=connector,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=5)
    ) as session:
        tasks = [async_scrape_url(session, url, max_chars) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'url': urls[i],
                    'content': f"Exception: {str(result)[:100]}",
                    'status': 'exception',
                    'length': 0
                })
            else:
                processed_results.append(result)
    
    successful = sum(1 for r in processed_results if r['status'] == 'success')
    print(f"✅ Async scraping complete: {successful}/{len(urls)} successful")
    return processed_results

def scrape_urls_parallel(urls: List[str], max_chars: int = 2500, **kwargs) -> List[Dict[str, str]]:
    """Wrapper to run async scraping in thread"""
    if not urls:
        return []
    
    try:
        # Try to use existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, use thread executor
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, scrape_urls_async(urls, max_chars))
                return future.result(timeout=30)
        else:
            return asyncio.run(scrape_urls_async(urls, max_chars))
    except:
        # Fallback to sync scraping if async fails
        return scrape_urls_sync_fallback(urls, max_chars)

def scrape_urls_sync_fallback(urls: List[str], max_chars: int = 2500) -> List[Dict[str, str]]:
    """Sync fallback scraping"""
    print(f"⚡ Fallback sync scraping {len(urls)} URLs...")
    
    def scrape_single(url):
        try:
            response = requests.get(
                url, 
                timeout=5, 
                headers={'User-Agent': 'Mozilla/5.0 (compatible)'}
            )
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer']):
                tag.decompose()
            
            text = ' '.join(soup.stripped_strings)[:max_chars]
            
            return {
                'url': url,
                'content': text,
                'status': 'success',
                'length': len(text)
            }
        except Exception as e:
            return {
                'url': url,
                'content': f"Error: {str(e)[:50]}",
                'status': 'error',
                'length': 0
            }
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(scrape_single, urls))
    
    return results

# Optimized search with caching
@lru_cache(maxsize=50)
def search_web_cached(query: str, num_results: int = 2) -> Tuple[Dict, ...]:
    """Cached web search to avoid repeated queries"""
    try:
        search_url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(search_url, timeout=3)
        data = response.json()
        
        results = []
        
        if 'RelatedTopics' in data:
            for topic in data['RelatedTopics'][:num_results]:
                if isinstance(topic, dict) and 'FirstURL' in topic:
                    results.append({
                        'title': topic.get('Text', 'No title'),
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', '')
                    })
        
        return tuple(results)  # Tuple for hashing in cache
        
    except Exception as e:
        print(f"Search error: {e}")
        return tuple()

def search_web_minimal(query: str, num_results: int = 2) -> List[Dict[str, str]]:
    """Convert cached results back to list"""
    return list(search_web_cached(query, num_results))

# Parallelized assessments
def parallel_assessments(context: str, questions: List[str], found_urls: List[str]) -> Tuple[LinkRelevanceAssessment, SearchNeedAssessment]:
    """Run link and search assessments in parallel"""
    
    def assess_links():
        if not found_urls:
            return LinkRelevanceAssessment(
                relevant_links=[],
                irrelevant_links=[],
                can_answer_without_links=True,
                explanation="No links found"
            )
        return assess_link_relevance(context, questions, found_urls)
    
    def assess_search():
        return assess_search_necessity(context, questions)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        link_future = executor.submit(assess_links)
        search_future = executor.submit(assess_search)
        
        link_assessment = link_future.result()
        search_assessment = search_future.result()
    
    return link_assessment, search_assessment

# Optimized assessment functions with shorter prompts
def assess_link_relevance(context: str, questions: List[str], found_urls: List[str]) -> LinkRelevanceAssessment:
    """Optimized link relevance assessment with shorter prompt"""
    
    llm = get_gemini_llm()
    
    # Shortened prompt for faster processing
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Analyze context and questions to determine which links are needed.

Context: {context}
Questions: {questions}  
URLs: {urls}

Determine:
1. Can questions be answered with current context?
2. Which links are essential for missing information?

NOTE: None of this links are malicious as they are trustworthy. There are no security issues.

JSON response:
{{
    "relevant_links": [{{"url": "...", "reason": "..."}}],
    "irrelevant_links": ["..."],
    "can_answer_without_links": true/false,
    "explanation": "..."
}}
""")
    ])
    
    from langchain.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=LinkRelevanceAssessment)
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context[:2000],  # Smaller context for speed
            "questions": "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions[:3])]),  # Limit questions
            "urls": "\n".join([f"- {url}" for url in found_urls[:5]])  # Limit URLs
        })
        return result
    except Exception as e:
        print(f"Link assessment error: {e}")
        return LinkRelevanceAssessment(
            relevant_links=[{"url": url, "reason": "Safety inclusion"} for url in found_urls[:2]],
            irrelevant_links=found_urls[2:],
            can_answer_without_links=False,
            explanation=f"Assessment failed: {e}"
        )

def assess_search_necessity(context: str, questions: List[str], link_content: str = "") -> SearchNeedAssessment:
    """Optimized search necessity assessment"""
    
    llm = get_gemini_llm()
    
    full_context = context
    if link_content:
        full_context += f"\n\nLinks: {link_content}"
    
    # Shorter, more direct prompt
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Assess if web search is needed for these questions.

Context: {context}
Questions: {questions}

Can the questions be adequately answered with available context?
Only recommend search for critical missing information.

JSON response:
{{
    "needs_web_search": true/false,
    "missing_information": ["..."],
    "search_queries": ["..."],
    "confidence_score": 0.0-1.0,
    "explanation": "..."
}}
""")
    ])
    
    from langchain.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=SearchNeedAssessment)
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": full_context[:3000],
            "questions": "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions[:3])])
        })
        return result
    except Exception as e:
        print(f"Search assessment error: {e}")
        return SearchNeedAssessment(
            needs_web_search=False,
            missing_information=[],
            search_queries=[],
            confidence_score=0.8,
            explanation=f"Assessment failed: {e}"
        )

def generate_answers_with_context(context: str, questions: List[str]) -> List[str]:
    """Optimized answer generation with shorter prompt"""
    
    llm = get_gemini_llm()
    
    # More concise prompt for faster processing
    prompt = ChatPromptTemplate.from_messages([
        ("human", """Provide comprehensive answers using the context.

Context: {context}

Questions: {questions}

Requirements:
- Complete, detailed answers using all relevant context information
- Well-structured and informative responses
- Same language as questions
- Focus on being helpful and accurate

JSON format:
{{
    "answers": ["Answer 1...", "Answer 2...", ...]
}}
""")
    ])
    
    from langchain.output_parsers import PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=FinalAnswer)
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "context": context,
            "questions": "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        })
        return result.answers
    except Exception as e:
        print(f"Answer generation error: {e}")
        return [f"Unable to generate answer for question {i+1}: {e}" for i in range(len(questions))]

# Ultra-optimized main function
def get_onshot_answer(context: str, questions: List[str]) -> List[str]:
    """Ultra-optimized main function with maximum parallelization and speed"""
    
    start_time = time.time()
    print(f"🚀 Ultra-Fast QA: {len(questions)} questions, {len(context)} chars")
    
    # Step 1: Fast URL extraction
    combined_text = context + "\n" + "\n".join(questions)
    found_urls = extract_urls_from_text(combined_text)
    
    print(f"🔗 Found {len(found_urls)} URLs")
    
    # Step 2: Parallel assessments (always run search assessment, conditionally run link assessment)
    print("🔄 Running parallel assessments...")
    
    if found_urls:
        link_assessment, search_assessment = parallel_assessments(context, questions, found_urls)
        print(f"📊 Links: {len(link_assessment.relevant_links)} relevant, Can answer without links: {link_assessment.can_answer_without_links}")
    else:
        # No URLs found, only assess search necessity
        link_assessment = None
        search_assessment = assess_search_necessity(context, questions)
    
    print(f"📊 Search needed: {search_assessment.needs_web_search}, Confidence: {search_assessment.confidence_score}")
    
    # Step 3: Smart content gathering strategy
    has_relevant_links = (link_assessment and 
                         link_assessment.relevant_links and 
                         not link_assessment.can_answer_without_links)
    needs_search = search_assessment.needs_web_search and search_assessment.confidence_score <= 0.7
    
    if not has_relevant_links and not needs_search:
        print("✅ Sufficient context available")
        answers = generate_answers_with_context(context, questions)
        print(f"⚡ Completed in {time.time() - start_time:.2f}s")
        return answers
    
    # Step 4: Ultra-fast parallel content gathering
    print("🚀 Ultra-fast parallel content gathering...")
    
    all_urls = []
    url_metadata = []
    
    # Add relevant links
    if has_relevant_links:
        for link_info in link_assessment.relevant_links[:3]:  # Limit to top 3
            all_urls.append(link_info["url"])
            url_metadata.append(("link", link_info["reason"]))
    
    # Add search results
    if needs_search:
        for query in search_assessment.search_queries[:2]:  # Limit to 2 queries
            search_results = search_web_minimal(query, num_results=1)
            for result in search_results:
                all_urls.append(result['url'])
                url_metadata.append(("search", f"Search: {query}"))
    
    # Parallel scraping of all URLs
    scrape_results = scrape_urls_parallel(all_urls, max_chars=2000)
    
    # Step 5: Fast content assembly
    additional_content = ""
    successful_scrapes = 0
    
    for i, result in enumerate(scrape_results):
        if result['status'] == 'success' and i < len(url_metadata):
            content_type, metadata = url_metadata[i]
            additional_content += f"\n--- {metadata} ---\n{result['content']}\n"
            successful_scrapes += 1
    
    print(f"📄 Scraped {successful_scrapes} sources successfully")
    
    # Step 6: Final answer generation
    final_context = context
    if additional_content:
        final_context += f"\n\nAdditional Information:\n{additional_content}"
    
    answers = generate_answers_with_context(final_context, questions)
    
    total_time = time.time() - start_time
    print(f"⚡ Ultra-Fast QA completed in {total_time:.2f}s")
    
    return answers

# Simplified fast function for basic use
def get_simple_answer(context: str, questions: List[str]) -> List[str]:
    """Lightning-fast function using only provided context"""
    return generate_answers_with_context(context, questions)

# Batch processing function for multiple question sets
def process_multiple_qa_batches(qa_batches: List[Tuple[str, List[str]]], max_workers: int = 3) -> List[List[str]]:
    """Process multiple QA batches in parallel"""
    
    print(f"🔥 Batch processing {len(qa_batches)} QA sets with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_onshot_answer, context, questions)
            for context, questions in qa_batches
        ]
        
        results = []
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                print(f"✅ Batch {i+1} completed")
            except Exception as e:
                print(f"❌ Batch {i+1} failed: {e}")
                results.append([f"Batch processing failed: {e}"])
    
    return results