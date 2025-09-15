import asyncio
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Any
import aiohttp
import json
import logging
import re
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedLLMOrchestrator:
    """Advanced orchestrator with caching, fallbacks, and monitoring"""

    def __init__(self):
        self.model_priority = {
            'analysis': ['llama3:latest', 'mixtral'],  # Use 8B instead of 70B
            'question_generation': ['mixtral', 'llama3:latest'],
            'evaluation': ['llama3:latest', 'mixtral']
            # 'analysis': ['llama3:70b', 'mixtral', 'llama3:latest'],
            # 'question_generation': ['mixtral', 'llama3:70b', 'llama3:latest'],
            # 'evaluation': ['llama3:70b', 'mixtral']
        }
        self.cache = {}
        self.session = None
        self.ollama_url = 'http://localhost:11434'

    async def initialize(self):
        """Initialize async session"""
        self.session = aiohttp.ClientSession()

    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def call_llm_with_fallback(self, prompt: str, purpose: str,
                                     max_tokens: int = 1000) -> Dict[str, Any]:
        """Call LLM with model fallback strategy and retry logic"""
        models = self.model_priority.get(purpose, ['mixtral'])
        last_error = None

        for model in models:
            try:
                logger.info(f"Trying model {model} for {purpose}")
                response = await self.session.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': model,
                        'prompt': prompt,
                        'stream': False,
                        'options': {
                            'temperature': 0.3 if purpose == 'evaluation' else 0.7,
                            'num_predict': max_tokens
                        }
                    },
                    timeout=180
                )

                if response.status == 200:
                    data = await response.json()
                    try:
                        # Try to parse the response as JSON
                        if 'response' in data:
                            response_text = data['response'].strip()

                            # Try to extract JSON from markdown code blocks
                            if response_text.startswith('```json'):
                                response_text = response_text[7:]
                            if response_text.endswith('```'):
                                response_text = response_text[:-3]
                            response_text = response_text.strip()

                            # Try to parse as JSON
                            try:
                                return json.loads(response_text)
                            except json.JSONDecodeError:
                                # If it's not valid JSON, return as text
                                logger.warning(
                                    f"Response is not valid JSON, returning as text: {response_text[:100]}...")
                                return {
                                    "response": response_text,
                                    "model_answer": response_text,
                                    "scoring_rubric": self._create_default_rubric(),
                                    "key_points": self._extract_key_points_from_answer(response_text),
                                    "red_flags": ["No specific answer provided", "Vague or unclear response"]
                                }
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        # Return a structured response even if JSON parsing fails
                        return {
                            "model_answer": data.get('response', 'No response generated'),
                            "scoring_rubric": self._create_default_rubric(),
                            "key_points": ["Understanding of core concepts", "Clear communication"],
                            "red_flags": ["No specific answer provided"]
                        }
                else:
                    error_text = await response.text()
                    last_error = f"Model {model} returned status {response.status}: {error_text}"
                    logger.error(last_error)

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_error = f"Model {model} failed for {purpose}: {str(e)}"
                logger.error(last_error)
                continue
            except Exception as e:
                last_error = f"Unexpected error with model {model}: {str(e)}"
                logger.error(last_error)
                continue

        raise Exception(f"All models failed for {purpose}. Last error: {last_error}")


    async def extract_jd_requirements(self, jd_text: str) -> Dict[str, Any]:
        """Extract and structure requirements from job description"""
        prompt = f"""
        Extract and categorize all requirements from this job description.

        JOB DESCRIPTION:
        {jd_text}

        Return a JSON object with these categories:
        - "technical_skills": list of technical skills with required proficiency levels
        - "soft_skills": list of soft skills/competencies
        - "responsibilities": key responsibilities and duties
        - "experience": required experience (years, types, industries)
        - "education_certifications": educational and certification requirements
        - "performance_metrics": how success would be measured in this role
        - "company_culture": cultural aspects and values mentioned

        For each item, include the source text from the JD that supports it.
        """

        try:
            result = await self.call_llm_with_fallback(prompt, 'analysis')

            # Extract JSON from the response if it's wrapped in text
            if isinstance(result, dict) and 'response' in result:
                response_text = result['response']
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from response")
                        # Fall through to return the original result

            # If we get here, return a default structure
            return {
                "technical_skills": [],
                "soft_skills": [],
                "responsibilities": [],
                "experience": [],
                "education_certifications": [],
                "performance_metrics": [],
                "company_culture": []
            }

        except Exception as e:
            logger.error(f"Error extracting JD requirements: {str(e)}")
            # Return a default structure on error
            return {
                "technical_skills": [],
                "soft_skills": [],
                "responsibilities": [],
                "experience": [],
                "education_certifications": [],
                "performance_metrics": [],
                "company_culture": []
            }

    async def generate_questions_cached(self, jd_hash: str, requirements: Dict) -> List[Dict]:
        """Generate questions with caching"""
        if jd_hash in self.cache:
            cache_entry = self.cache[jd_hash]
            if datetime.now() - cache_entry['timestamp'] < timedelta(hours=24):
                return cache_entry['questions']

        questions = await self._generate_questions(requirements)
        self.cache[jd_hash] = {
            'questions': questions,
            'timestamp': datetime.now()
        }
        return questions

    async def _generate_questions(self, requirements: Dict) -> List[Dict]:
        # Extract just the technical skills for question generation
        technical_skills = requirements.get('technical_skills', [])

        # Use .get() method to safely access dictionary keys
        skills_text = "\n".join([f"- {skill.get('skill', 'N/A')}: {skill.get('proficiency', '')}"
                                 for skill in technical_skills])

        prompt = f"""
                Generate 8-12 interview questions based on these technical skills:

                {skills_text}

                Create questions with this format:
                - 40% technical questions (specific to the skills listed)
                - 30% behavioral questions (about past experiences)
                - 20% situational questions (hypothetical scenarios)
                - 10% culture fit questions (values and work style)

                For each question, provide a JSON object with:
                - "question_id": unique ID like "Q1", "Q2"
                - "question_text": the actual question
                - "question_type": technical, behavioral, situational, or cultural
                - "difficulty": easy, medium, or hard
                - "skill_tested": which specific skill this question tests

                Return ONLY a JSON array of these question objects, without any additional text.
                """

        try:
            result = await self.call_llm_with_fallback(prompt, 'question_generation')

            # Handle both direct JSON and wrapped responses
            if isinstance(result, dict) and 'response' in result:
                try:
                    # Try to parse the response as JSON
                    questions = json.loads(result['response'])
                    if isinstance(questions, list):
                        return questions
                    else:
                        logger.error(f"Expected a list of questions but got: {type(questions)}")
                        return []
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON from response: {e}")
                    return []
            elif isinstance(result, list):
                return result
            else:
                logger.error(f"Unexpected result type: {type(result)}")
                return []

        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    async def generate_answer_guidelines(self, question: Dict, requirements: Dict) -> Dict[str, Any]:
        """Generate answer guidelines for a question with fallback values"""
        prompt = f"""
        Create comprehensive answer guidelines for this interview question:

        QUESTION: {question.get('question_text', '')}

        JOB REQUIREMENTS:
        {json.dumps(requirements, indent=2)}

        Provide:
        1. A model answer that would be considered excellent
        2. A scoring rubric (1-5 scale) with specific criteria for each level
        3. Key points that must be addressed in a good answer
        4. Common mistakes or red flags to watch for

        Return as JSON with:
        - "model_answer": comprehensive ideal answer
        - "scoring_rubric": detailed criteria for scores 1 through 5
        - "key_points": list of specific points that must be addressed
        - "red_flags": list of concerning responses or omissions
        """

        try:
            result = await self.call_llm_with_fallback(prompt, 'evaluation')

            # Parse the response to ensure it has the required structure
            guidelines = self._parse_guidelines_response(result)
            return guidelines

        except Exception as e:
            logger.error(f"Error generating answer guidelines: {str(e)}")
            # Return fallback structure if LLM fails
            return self._get_fallback_guidelines(question)

    def _get_fallback_guidelines(self, question: Dict) -> Dict[str, Any]:
        """Get fallback guidelines when LLM fails"""
        return {
            "model_answer": f"A comprehensive answer addressing {question.get('question_text', 'the question')}",
            "scoring_rubric": self._create_default_rubric(),
            "key_points": [
                "Understanding of core concepts",
                "Clear communication",
                "Relevant examples",
                "Structured thinking",
                "Practical application"
            ],
            "red_flags": [
                "No specific answer provided",
                "Vague or unclear response",
                "Lack of relevant examples",
                "Inability to explain concepts clearly"
            ]
        }

    def _parse_guidelines_response(self, result: Any) -> Dict[str, Any]:
        """Parse and validate the guidelines response"""
        guidelines = {}

        # Handle different response formats
        if isinstance(result, dict):
            if 'response' in result:
                try:
                    # Try to parse the response as JSON
                    response_data = json.loads(result['response'])
                    guidelines = response_data
                except (json.JSONDecodeError, TypeError):
                    # If it's not JSON, use the response as model_answer
                    guidelines = {
                        'model_answer': result['response'],
                        'scoring_rubric': self._create_default_rubric(),
                        'key_points': self._extract_key_points_from_answer(result['response']),
                        'red_flags': ["No specific answer provided", "Vague or unclear response"]
                    }
            else:
                guidelines = result
        else:
            # If result is not a dictionary, create a basic structure
            guidelines = {
                'model_answer': str(result),
                'scoring_rubric': self._create_default_rubric(),
                'key_points': self._extract_key_points_from_answer(str(result)),
                'red_flags': ["No specific answer provided", "Vague or unclear response"]
            }

        # Ensure all required keys exist
        required_keys = ['model_answer', 'scoring_rubric', 'key_points', 'red_flags']
        for key in required_keys:
            if key not in guidelines:
                if key == 'scoring_rubric':
                    guidelines[key] = self._create_default_rubric()
                elif key == 'key_points':
                    guidelines[key] = self._extract_key_points_from_answer(guidelines.get('model_answer', ''))
                elif key == 'red_flags':
                    guidelines[key] = ["No specific answer provided", "Vague or unclear response"]
                else:
                    guidelines[key] = "Not available"

        return guidelines

    def _extract_key_points_from_answer(self, answer: str) -> List[str]:
        """Extract key points from a model answer"""
        if not answer:
            return ["Understanding of core concepts", "Clear communication", "Relevant examples"]

        # Simple extraction of key points by splitting the answer
        sentences = re.split(r'[.!?]+', answer)
        key_points = [s.strip() for s in sentences if len(s.strip()) > 10][:5]  # Take up to 5 meaningful sentences
        return key_points if key_points else ["Understanding of core concepts", "Clear communication",
                                              "Relevant examples"]

    def _create_default_rubric(self) -> Dict[str, str]:
        """Create a default scoring rubric"""
        return {
            "1": "No relevant answer or completely incorrect",
            "2": "Partial answer with major inaccuracies",
            "3": "Basic understanding with some inaccuracies",
            "4": "Good answer with minor omissions",
            "5": "Comprehensive and accurate answer"
        }

    async def generate_interview_guide(self, questions: List[Dict], requirements: Dict) -> str:
        if not questions:
            return "Interview guide generation failed - no questions available"

        prompt = f"""
        Create a concise interview guide based on these questions and requirements.

        TECHNICAL SKILLS REQUIRED:
        {json.dumps(requirements.get('technical_skills', []), indent=2)}

        QUESTIONS TO ASK:
        {json.dumps(questions, indent=2)}

        The guide should include:
        1. Brief introduction to the interview structure
        2. Suggested time allocation per question type
        3. Key things to listen for in responses
        4. Evaluation criteria overview

        Format the guide in clear, concise markdown without unnecessary fluff.
        """

        result = await self.call_llm_with_fallback(prompt, 'analysis')
        return result if isinstance(result, str) else "Interview guide generation failed"