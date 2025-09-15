import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Any
import json


class QuestionQualityEngine:
    """Ensures question quality, diversity, and relevance"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

    def calculate_question_diversity(self, questions: List[Dict]) -> float:
        """Ensure questions aren't redundant"""
        question_texts = [q.get('question_text', '') for q in questions]

        if len(question_texts) < 2:
            return 1.0

        tfidf_matrix = self.vectorizer.fit_transform(question_texts)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(similarity_matrix, 0)

        avg_similarity = np.mean(similarity_matrix)
        return 1.0 - avg_similarity

    def validate_question_quality(self, question: Dict, jd_requirements: Dict) -> Dict[str, Any]:
        """Comprehensive question validation"""
        question_text = question.get('question_text', '')

        validation_results = {
            'relevance_score': self._calculate_relevance(question, jd_requirements),
            'clarity_score': self._calculate_clarity(question_text),
            'specificity_score': self._calculate_specificity(question),
            'bias_potential': self._check_for_bias(question_text),
            'action_verb_presence': self._check_action_verbs(question_text),
            'open_endedness': self._check_open_ended(question_text)
        }

        validation_results['overall_score'] = (
                validation_results['relevance_score'] * 0.4 +
                validation_results['clarity_score'] * 0.3 +
                validation_results['specificity_score'] * 0.3
        )

        return validation_results

    def _calculate_relevance(self, question: Dict, jd_requirements: Dict) -> float:
        """Calculate how relevant the question is to JD requirements"""
        # Extract all text from requirements for comparison
        requirements_text = " ".join([
            " ".join(str(v) for v in jd_requirements.values())
            if isinstance(v, (list, dict)) else str(v)
            for v in jd_requirements.values()
        ]).lower()

        question_text = question.get('question_text', '').lower()

        # Count keyword matches with more relevant terms
        keywords = ['experience', 'skill', 'ability', 'knowledge', 'responsible',
                    'develop', 'design', 'implement', 'manage', 'lead', 'create']
        matches = sum(1 for keyword in keywords if keyword in question_text and keyword in requirements_text)

        return min(matches / len(keywords), 1.0)

    def _calculate_clarity(self, question_text: str) -> float:
        """Calculate question clarity"""
        word_count = len(question_text.split())
        sentence_count = len(re.findall(r'[.!?]+', question_text))

        if word_count > 30 or sentence_count > 2:
            return 0.6  # Too long or complex
        elif word_count < 5:
            return 0.7  # Too short

        return 0.9  # Good length

    def _calculate_specificity(self, question: Dict) -> float:
        """Calculate question specificity"""
        question_text = question.get('question_text', '')

        # Check for specific context references
        specific_indicators = ['describe a time', 'give an example', 'specific', 'detailed',
                               'situation where', 'tell me about a time']
        indicators_present = sum(1 for indicator in specific_indicators
                                 if indicator in question_text.lower())

        return min(indicators_present / len(specific_indicators), 1.0)

    def _check_for_bias(self, question_text: str) -> str:
        """Check for potential bias in question"""
        bias_indicators = {
            'gender': ['he', 'she', 'him', 'her', 'male', 'female', 'man', 'woman'],
            'age': ['young', 'old', 'recent graduate', 'seasoned', 'fresh', 'experienced'],
            'family': ['marriage', 'children', 'parent', 'pregnancy', 'family plans'],
            'culture': ['nationality', 'ethnicity', 'religion', 'cultural background']
        }

        for bias_type, indicators in bias_indicators.items():
            for indicator in indicators:
                if indicator in question_text.lower():
                    return f"Potential {bias_type} bias detected"

        return "No obvious bias detected"

    def _check_action_verbs(self, question_text: str) -> bool:
        """Check if question uses action verbs"""
        action_verbs = ['describe', 'explain', 'analyze', 'evaluate', 'create', 'design',
                        'develop', 'implement', 'manage', 'lead', 'solve', 'improve']
        return any(verb in question_text.lower() for verb in action_verbs)

    def _check_open_ended(self, question_text: str) -> bool:
        """Check if question is open-ended"""
        closed_indicators = ['is', 'are', 'do', 'does', 'have', 'has', 'did you', 'can you']
        return not any(question_text.strip().lower().startswith(indicator)
                       for indicator in closed_indicators)