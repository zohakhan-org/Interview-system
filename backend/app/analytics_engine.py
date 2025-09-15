import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import json


class InterviewAnalyticsEngine:
    """Advanced analytics and insights for interview process"""

    def __init__(self):
        self.interview_data = pd.DataFrame()
        self.kit_generation_data = []

    def _clean_json_data(self, data: Any) -> Any:
        """
        Recursively clean data to ensure JSON compatibility by replacing
        NaN, Inf, and -Inf values with None or appropriate replacements
        """
        if isinstance(data, dict):
            return {k: self._clean_json_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_json_data(item) for item in data]
        elif isinstance(data, float):
            # Replace non-finite values with None
            if pd.isna(data) or np.isinf(data) or np.isneginf(data):
                return None
            return data
        else:
            return data

    def record_kit_generation(self, kit: Dict):
        """Record kit generation for analytics"""
        self.kit_generation_data.append({
            'timestamp': datetime.now(),
            'question_count': len(kit.get('questions', [])),
            'jd_hash': kit.get('jd_hash', ''),
            'requirements_categories': len(kit.get('job_requirements', {}))
        })

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report with JSON-compliant data"""
        if not self.kit_generation_data:
            return {"message": "No data available yet"}

        df = pd.DataFrame(self.kit_generation_data)

        report = {
            'summary_metrics': self._calculate_summary_metrics(df),
            'time_analysis': self._analyze_time_metrics(df),
            'recent_activity': self._get_recent_activity(df)
        }

        # Clean the report data to ensure JSON compatibility
        return self._clean_json_data(report)

    def _calculate_summary_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key summary metrics with NaN handling"""
        if df.empty:
            return {
                'total_kits_generated': 0,
                'average_questions_per_kit': 0.0,
                'average_categories_per_jd': 0.0,
                'kits_last_7_days': 0
            }

        # Convert numpy types to native Python types with NaN handling
        total_kits = len(df)
        avg_questions = float(df['question_count'].mean()) if not df.empty else 0.0
        avg_categories = float(df['requirements_categories'].mean()) if not df.empty else 0.0

        # Calculate kits in last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_data = df[df['timestamp'] > seven_days_ago]
        kits_last_7_days = int(len(recent_data)) if not recent_data.empty else 0

        # Ensure no NaN values
        avg_questions = 0.0 if pd.isna(avg_questions) else avg_questions
        avg_categories = 0.0 if pd.isna(avg_categories) else avg_categories

        return {
            'total_kits_generated': int(total_kits),
            'average_questions_per_kit': float(avg_questions),
            'average_categories_per_jd': float(avg_categories),
            'kits_last_7_days': int(kits_last_7_days)
        }

    def _analyze_time_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time-based metrics with NaN handling"""
        if df.empty:
            return {
                'daily_average': 0.0,
                'daily_std': 0.0,
                'busiest_day': None,
                'busiest_day_count': 0
            }

        df['date'] = df['timestamp'].dt.date
        daily_counts = df.groupby('date').size()

        # Convert numpy types to native Python types with NaN handling
        daily_avg = float(daily_counts.mean()) if not daily_counts.empty else 0.0
        daily_std = float(daily_counts.std()) if not daily_counts.empty else 0.0

        # Handle potential NaN values
        daily_avg = 0.0 if pd.isna(daily_avg) else daily_avg
        daily_std = 0.0 if pd.isna(daily_std) else daily_std

        busiest_day = daily_counts.idxmax() if not daily_counts.empty else None
        busiest_count = int(daily_counts.max()) if not daily_counts.empty else 0

        return {
            'daily_average': daily_avg,
            'daily_std': daily_std,
            'busiest_day': busiest_day.isoformat() if busiest_day else None,
            'busiest_day_count': busiest_count
        }

    def _get_recent_activity(self, df: pd.DataFrame) -> List[Dict]:
        """Get recent activity data with NaN handling"""
        recent = df[df['timestamp'] > datetime.now() - timedelta(hours=24)]
        if recent.empty:
            return []

        # Convert DataFrame to list of dictionaries with native Python types
        result = []
        for _, row in recent.iterrows():
            result.append({
                'timestamp': row['timestamp'].isoformat(),
                'question_count': int(row['question_count']),
                'jd_hash': row['jd_hash'],
                'requirements_categories': int(row['requirements_categories'])
            })

        return result