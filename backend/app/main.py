from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import hashlib
from datetime import datetime
from typing import Dict, List, Any
import aiohttp
import asyncio
import logging
import json
# Import components
from .advanced_llm_orchestrator import AdvancedLLMOrchestrator
from .question_quality_engine import QuestionQualityEngine
from .advanced_evaluation_engine import AdvancedEvaluationEngine
from .collaboration_engine import InterviewCollaborationEngine
from .analytics_engine import InterviewAnalyticsEngine


def validate_question_structure(question: Dict) -> Dict:
    """Ensure the question has all required fields"""
    required_fields = ['question_id', 'question_text', 'question_type', 'difficulty']

    for field in required_fields:
        if field not in question:
            if field == 'question_id':
                question['question_id'] = f"Q{hash(question.get('question_text', '')) % 10000}"
            elif field == 'question_text':
                question['question_text'] = "Question text not available"
            elif field == 'question_type':
                question['question_type'] = "technical"
            elif field == 'difficulty':
                question['difficulty'] = "medium"

    return question

app = FastAPI(title="Enterprise JD-Centric Interview System", version="1.0.0")
logger = logging.getLogger(__name__)
# Initialize components
llm_orchestrator = AdvancedLLMOrchestrator()
quality_engine = QuestionQualityEngine()
evaluation_engine = AdvancedEvaluationEngine()
collaboration_engine = InterviewCollaborationEngine()
analytics_engine = InterviewAnalyticsEngine()

# Store generated kits to avoid regeneration in evaluate-response
generated_kits = {}



class JobDescription(BaseModel):
    text: str


class CandidateResponse(BaseModel):
    question_id: str
    response: str

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            # Handle NaN and Infinity values
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return super().default(obj)

# Override the default JSON encoder in your FastAPI app
app = FastAPI(title="Enterprise JD-Centric Interview System", version="1.0.0")
app.json_encoder = CustomJSONEncoder

class InterviewFeedback(BaseModel):
    session_id: str
    user_id: str
    question_id: str
    score: int
    comments: str


@app.on_event("startup")
async def startup_event():
    await llm_orchestrator.initialize()


@app.on_event("shutdown")
async def shutdown_event():
    await llm_orchestrator.close()


@app.post("/api/generate-interview-kit")
async def generate_interview_kit(jd: JobDescription, background_tasks: BackgroundTasks):
    try:
        jd_hash = hashlib.md5(jd.text.encode()).hexdigest()

        if jd_hash in generated_kits:
            return generated_kits[jd_hash]

        # Extract requirements from JD
        requirements = await llm_orchestrator.extract_jd_requirements(jd.text)

        # Generate questions - with fallback if empty
        questions = await llm_orchestrator.generate_questions_cached(jd_hash, requirements)
        if not questions:
            logger.warning("Question generation failed, using fallback questions")
            questions = generate_fallback_questions(requirements)

        # Enhance questions with quality metrics and answer guidelines
        enhanced_questions = []
        for question in questions:
            # Validate question structure
            question = validate_question_structure(question)

            quality_metrics = quality_engine.validate_question_quality(question, requirements)
            question['quality_metrics'] = quality_metrics

            # Generate answer guidelines with error handling
            try:
                guidelines = await llm_orchestrator.generate_answer_guidelines(question, requirements)
                question['answer_guidelines'] = guidelines
            except Exception as e:
                logger.error(
                    f"Failed to generate guidelines for question {question.get('question_id', 'unknown')}: {str(e)}")
                question['answer_guidelines'] = {
                    "model_answer": f"A comprehensive answer addressing {question.get('question_text', 'the question')}",
                    "scoring_rubric": {
                        "1": "No relevant answer or completely incorrect",
                        "2": "Partial answer with major inaccuracies",
                        "3": "Basic understanding with some inaccuracies",
                        "4": "Good answer with minor omissions",
                        "5": "Comprehensive and accurate answer"
                    },
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

            enhanced_questions.append(question)

        # Generate interview guide
        guide = await llm_orchestrator.generate_interview_guide(enhanced_questions, requirements)

        kit = {
            "job_requirements": requirements,
            "questions": enhanced_questions,
            "interview_guide": guide,
            "generated_at": datetime.now().isoformat(),
            "jd_hash": jd_hash
        }

        generated_kits[jd_hash] = kit
        background_tasks.add_task(analytics_engine.record_kit_generation, kit)

        return kit

    except Exception as e:
        logger.error(f"Failed to generate interview kit: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate interview kit: {str(e)}")
def _validate_question_structure(question: Dict) -> Dict:
    """Ensure the question has all required fields"""
    required_fields = ['question_id', 'question_text', 'question_type', 'difficulty']

    for field in required_fields:
        if field not in question:
            if field == 'question_id':
                question['question_id'] = f"Q{hash(question.get('question_text', '')) % 10000}"
            elif field == 'question_text':
                question['question_text'] = "Question text not available"
            elif field == 'question_type':
                question['question_type'] = "technical"
            elif field == 'difficulty':
                question['difficulty'] = "medium"

    return question

def generate_fallback_questions(requirements: Dict) -> List[Dict]:
    """Generate basic fallback questions if LLM fails"""
    technical_skills = requirements.get('technical_skills', [])
    questions = []

    # Technical questions
    for i, skill in enumerate(technical_skills[:5]):  # Limit to 5 skills
        # Safely access skill information
        skill_name = skill.get('skill', 'technical skills') if isinstance(skill, dict) else str(skill)
        questions.append({
            "question_id": f"Q{i + 1}",
            "question_text": f"What experience do you have with {skill_name}?",
            "question_type": "technical",
            "difficulty": "medium",
            "skill_tested": skill_name
        })

    # Behavioral questions
    behavioral_questions = [
        "Tell me about a challenging project you worked on and how you approached it.",
        "Describe a time when you had to learn a new technology quickly.",
        "How do you handle disagreements with team members about technical approaches?",
        "Can you give an example of how you've mentored or helped a junior developer?"
    ]

    for i, question in enumerate(behavioral_questions, start=len(questions) + 1):
        questions.append({
            "question_id": f"Q{i}",
            "question_text": question,
            "question_type": "behavioral",
            "difficulty": "medium",
            "skill_tested": "Problem-solving and collaboration"
        })

    return questions
@app.post("/api/evaluate-response")
async def evaluate_response(response: CandidateResponse, jd: JobDescription):
    """Evaluate a candidate's response to a specific question"""
    try:
        # Create a hash of the JD for lookup
        jd_hash = hashlib.md5(jd.text.encode()).hexdigest()

        # Get the kit from cache or generate it
        if jd_hash not in generated_kits:
            # If not in cache, generate it
            kit = await generate_interview_kit(jd, BackgroundTasks())
        else:
            kit = generated_kits[jd_hash]

        # Find the question
        question = None
        for q in kit['questions']:
            if q.get('question_id') == response.question_id:
                question = q
                break

        if not question:
            raise HTTPException(status_code=404, detail="Question not found")

        # Evaluate the response
        evaluation = evaluation_engine.evaluate_response(
            response.response,
            question['answer_guidelines']['model_answer'],
            question['answer_guidelines']['key_points']
        )

        return {
            "question": question['question_text'],
            "evaluation": evaluation,
            "ideal_answer_highlights": question['answer_guidelines']['key_points']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to evaluate response: {str(e)}")


@app.post("/api/create-interview-session")
async def create_interview_session(candidate_id: str, jd: JobDescription):
    """Create a new collaborative interview session"""
    try:
        kit = await generate_interview_kit(jd, BackgroundTasks())
        session_id = await collaboration_engine.create_interview_session(candidate_id, kit)
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/api/submit-feedback")
async def submit_feedback(feedback: InterviewFeedback):
    """Submit feedback for an interview question"""
    try:
        success = await collaboration_engine.submit_feedback(
            feedback.session_id,
            feedback.user_id,
            feedback.question_id,
            {"score": feedback.score, "comments": feedback.comments}
        )

        if success:
            return {"status": "success"}
        else:
            raise HTTPException(status_code=404, detail="Session or question not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview"""
    try:
        return analytics_engine.generate_performance_report()
    except Exception as e:
        logger.error(f"Error generating analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate analytics: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "llm_orchestrator": "active",
            "quality_engine": "active",
            "evaluation_engine": "active",
            "collaboration_engine": "active",
            "analytics_engine": "active"
        }
    }


@app.get("/api/health/detailed")
async def detailed_health_check():
    """Detailed health check with Ollama connectivity"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:11434/api/tags', timeout=180) as response:
                ollama_healthy = response.status == 200
    except:
        ollama_healthy = False

    return {
        "status": "healthy" if ollama_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "ollama_connected": ollama_healthy,
        "active_models": await get_available_models() if ollama_healthy else []
    }


async def get_available_models():
    """Get available models from Ollama"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:11434/api/tags') as response:
                data = await response.json()
                return [model['name'] for model in data.get('models', [])]
    except:
        return []

@app.post("/api/test-ollama")
async def test_ollama():
    """Test endpoint for Ollama integration"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3:70b',
                    'prompt': 'Why is the sky blue?',
                    'stream': False
                },
                timeout=180
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"status": "success", "response": data}
                else:
                    error_text = await response.text()
                    return {"status": "error", "message": f"Ollama returned status {response.status}: {error_text}"}
    except aiohttp.ClientError as e:
        return {"status": "error", "message": f"Client error: {str(e)}"}
    except asyncio.TimeoutError:
        return {"status": "error", "message": "Request to Ollama timed out"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)