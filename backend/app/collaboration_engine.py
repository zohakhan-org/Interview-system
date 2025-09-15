import asyncio
from typing import Dict, List, Any
from datetime import datetime
import json


class InterviewCollaborationEngine:
    """Real-time collaboration for interview teams"""

    def __init__(self):
        self.active_sessions = {}
        self.feedback_store = {}

    async def create_interview_session(self, candidate_id: str, interview_kit: Dict) -> str:
        """Create a new collaborative interview session"""
        session_id = f"session_{candidate_id}_{datetime.now().timestamp()}"

        self.active_sessions[session_id] = {
            'candidate_id': candidate_id,
            'interview_kit': interview_kit,
            'participants': {},
            'current_question': None,
            'feedback': {},
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }

        return session_id

    async def add_participant(self, session_id: str, user_id: str, user_role: str) -> bool:
        """Add participant to interview session"""
        if session_id not in self.active_sessions:
            return False

        self.active_sessions[session_id]['participants'][user_id] = {
            'role': user_role,
            'joined_at': datetime.now(),
            'last_active': datetime.now()
        }

        return True

    async def submit_feedback(self, session_id: str, user_id: str, question_id: str, feedback: Dict) -> bool:
        """Submit feedback for a question"""
        if session_id not in self.active_sessions:
            return False

        if 'feedback' not in self.active_sessions[session_id]:
            self.active_sessions[session_id]['feedback'] = {}

        if question_id not in self.active_sessions[session_id]['feedback']:
            self.active_sessions[session_id]['feedback'][question_id] = {}

        self.active_sessions[session_id]['feedback'][question_id][user_id] = {
            **feedback,
            'submitted_at': datetime.now()
        }

        # Calculate consensus score
        self._calculate_consensus(session_id, question_id)

        return True

    def _calculate_consensus(self, session_id: str, question_id: str):
        """Calculate consensus among interviewers"""
        if session_id not in self.active_sessions:
            return

        feedbacks = self.active_sessions[session_id].get('feedback', {}).get(question_id, {})

        # Remove the consensus entry if it exists to avoid including it in scores
        feedbacks.pop('consensus', None)

        if not feedbacks:
            return

        scores = [fb.get('score', 0) for fb in feedbacks.values() if isinstance(fb, dict)]
        if not scores:
            return

        avg_score = sum(scores) / len(scores)

        # Store consensus data
        if 'feedback' not in self.active_sessions[session_id]:
            self.active_sessions[session_id]['feedback'] = {}

        if question_id not in self.active_sessions[session_id]['feedback']:
            self.active_sessions[session_id]['feedback'][question_id] = {}

        self.active_sessions[session_id]['feedback'][question_id]['consensus'] = {
            'average_score': avg_score,
            'score_range': (min(scores), max(scores)),
            'participant_count': len(scores),
            'calculated_at': datetime.now()
        }

    async def get_session_consensus(self, session_id: str) -> Dict[str, Any]:
        """Get consensus data for a session"""
        if session_id not in self.active_sessions:
            return {}

        consensus_data = {}
        for question_id, feedback in self.active_sessions[session_id].get('feedback', {}).items():
            if 'consensus' in feedback:
                consensus_data[question_id] = feedback['consensus']

        return consensus_data