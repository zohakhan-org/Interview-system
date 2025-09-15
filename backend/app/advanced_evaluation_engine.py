import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Any


class AdvancedEvaluationEngine:
    """Sophisticated candidate response evaluation with multiple metrics"""

    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_weights = {
            'technical_terms': 1.5,
            'action_verbs': 1.3,
            'quantifiable_results': 2.0,
            'problem_solving_indicators': 1.7
        }

    def evaluate_response(self, response: str, ideal_answer: str,
                          evaluation_criteria: List[str]) -> Dict[str, Any]:
        """Comprehensive response evaluation"""
        evaluation = {
            'semantic_similarity': self._calculate_semantic_similarity(response, ideal_answer),
            'keyword_coverage': self._calculate_keyword_coverage(response, ideal_answer, evaluation_criteria),
            'structured_thinking': self._detect_structured_thinking(response),
            'conciseness_score': self._calculate_conciseness(response),
            'examples_quality': self._evaluate_examples_quality(response),
            'criteria_alignment': self._check_criteria_alignment(response, evaluation_criteria)
        }

        # Calculate overall score with weighted average
        weights = {
            'semantic_similarity': 0.25,
            'keyword_coverage': 0.20,
            'structured_thinking': 0.20,
            'conciseness_score': 0.10,
            'examples_quality': 0.15,
            'criteria_alignment': 0.10
        }

        evaluation['overall_score'] = sum(
            evaluation[metric] * weights[metric] for metric in weights
        )

        return evaluation

    def _calculate_semantic_similarity(self, response: str, ideal_answer: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            response_embedding = self.embedding_model.encode([response])
            ideal_embedding = self.embedding_model.encode([ideal_answer])
            return cosine_similarity(response_embedding, ideal_embedding)[0][0]
        except Exception:
            return 0.0  # Fallback if encoding fails

    def _calculate_keyword_coverage(self, response: str, ideal_answer: str, criteria: List[str]) -> float:
        """Calculate keyword coverage based on evaluation criteria"""
        if not criteria:
            return 0.5

        # Extract meaningful keywords from criteria
        keywords = set()
        for criterion in criteria:
            # Get words longer than 4 characters for better matching
            words = re.findall(r'\b[a-zA-Z]{5,}\b', criterion.lower())
            keywords.update(words)

        if not keywords:
            return 0.5

        response_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', response.lower()))
        matches = keywords.intersection(response_words)
        return len(matches) / len(keywords)

    def _detect_structured_thinking(self, response: str) -> float:
        """Detect structured thinking patterns"""
        structure_indicators = [
            'first', 'next', 'then', 'finally', 'step',
            'framework', 'approach', 'identified', 'analyzed',
            'implemented', 'evaluated', 'methodology', 'process'
        ]

        indicator_count = sum(1 for indicator in structure_indicators
                              if indicator in response.lower())
        return min(indicator_count / 5, 1.0)

    def _calculate_conciseness(self, response: str) -> float:
        """Calculate response conciseness"""
        word_count = len(response.split())

        if word_count < 30:
            return 0.7  # Possibly too short
        elif word_count > 300:
            return 0.6  # Too verbose
        elif word_count > 150:
            return 0.8  # Slightly long
        else:
            return 0.9  # Good length

    def _evaluate_examples_quality(self, response: str) -> float:
        """Evaluate quality of examples provided"""
        example_indicators = ['for example', 'for instance', 'such as', 'in my experience', 'example']
        has_examples = any(indicator in response.lower() for indicator in example_indicators)

        # Check for specific details in examples
        detail_indicators = ['number', 'percent', 'result', 'outcome', 'impact', 'increased', 'decreased', 'saved']
        has_details = any(indicator in response.lower() for indicator in detail_indicators)

        if has_examples and has_details:
            return 0.9
        elif has_examples:
            return 0.7
        else:
            return 0.5

    def _check_criteria_alignment(self, response: str, criteria: List[str]) -> float:
        """Check alignment with evaluation criteria"""
        if not criteria:
            return 0.5

        # Check if response addresses the criteria
        matches = 0
        response_lower = response.lower()

        for criterion in criteria:
            # Extract key terms from each criterion
            key_terms = re.findall(r'\b[a-zA-Z]{4,}\b', criterion.lower())
            if any(term in response_lower for term in key_terms):
                matches += 1

        return matches / len(criteria)