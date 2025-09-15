import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

# Configuration - Use environment variable for backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
st.set_page_config(page_title="AI Interview System", layout="wide")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "interview_kit" not in st.session_state:
    st.session_state.interview_kit = None
if "current_question_index" not in st.session_state:
    st.session_state.current_question_index = 0
if "debug" not in st.session_state:
    st.session_state.debug = False
if "download_data" not in st.session_state:
    st.session_state.download_data = None


def get_local_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return image_file.read()
    except FileNotFoundError:
        st.error(f"Image file not found: {image_path}")
        return None


def call_backend(endpoint, method="GET", data=None):
    """Helper function to call backend API"""
    try:
        url = f"{BACKEND_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=180)
        elif method == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=180)

        response.raise_for_status()  # This will raise an exception for 4xx/5xx responses
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend server. Please make sure it's running on port 8000.")
        return None
    except requests.exceptions.Timeout:
        st.error("The request to the backend server timed out. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"Backend server returned an error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None


def safe_get(obj, key, default=None):
    """Safely get a value from a dictionary with fallback"""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def validate_question_structure(question):
    """Ensure a question has all required fields with default values"""
    if not isinstance(question, dict):
        return {
            "question_id": "invalid",
            "question_text": "Invalid question structure",
            "question_type": "technical",
            "difficulty": "medium",
            "skill_tested": "Unknown"
        }

    return {
        "question_id": question.get("question_id", f"Q{hash(str(question)) % 10000}"),
        "question_text": question.get("question_text", "Question text not available"),
        "question_type": question.get("question_type", "technical"),
        "difficulty": question.get("difficulty", "medium"),
        "skill_tested": question.get("skill_tested", "Unknown skill"),
        "quality_metrics": question.get("quality_metrics", {}),
        "answer_guidelines": question.get("answer_guidelines", {})
    }


def main():
    st.sidebar.title("AI Interview System")
    image_path = "static/interview.jpeg"
    image_data = get_local_image(image_path)
    if image_data:
        st.sidebar.image(image_data, width=64)
    else:
        # Fallback to text if image not available
        st.sidebar.markdown("🎤 **Interview System**")

    # Debug toggle
    st.session_state.debug = st.sidebar.checkbox("Debug Mode", value=st.session_state.debug)

    menu = st.sidebar.selectbox("Navigation", [
        "Dashboard",
        "Generate Interview Kit",
        "Conduct Interview",
        "Evaluate Responses",
        "Analytics & Reports"
    ])

    if menu == "Dashboard":
        show_dashboard()
    elif menu == "Generate Interview Kit":
        generate_interview_kit()
    elif menu == "Conduct Interview":
        conduct_interview()
    elif menu == "Evaluate Responses":
        evaluate_responses()
    elif menu == "Analytics & Reports":
        show_analytics()


def show_dashboard():
    st.title("🎯 Interview System Dashboard")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Kits Generated", "24", "5 this week")
    with col2:
        st.metric("Avg. Candidate Score", "7.8/10", "+0.5 from last month")
    with col3:
        st.metric("Interview Completion", "92%", "4% improvement")

    st.divider()

    # Quick actions
    st.subheader("Quick Actions")
    action_col1, action_col2, action_col3 = st.columns(3)

    with action_col1:
        if st.button("🔄 Generate New Kit", width="stretch"):
            st.session_state.current_page = "Generate Interview Kit"
            st.rerun()

    with action_col2:
        if st.button("🎤 Start New Interview", width="stretch"):
            st.session_state.current_page = "Conduct Interview"
            st.rerun()

    with action_col3:
        if st.button("📊 View Analytics", width="stretch"):
            st.session_state.current_page = "Analytics & Reports"
            st.rerun()

    # Recent activity
    st.subheader("Recent Activity")
    activity_data = {
        "Time": ["2 hours ago", "5 hours ago", "1 day ago", "2 days ago"],
        "Activity": ["Interview Kit Generated", "Candidate Evaluated", "Interview Completed", "New JD Processed"],
        "Details": ["Senior Data Scientist", "John Doe scored 8.2/10", "3 participants, 92% consensus",
                    "Machine Learning Engineer"]
    }

    st.dataframe(activity_data, width="stretch", hide_index=True)


def generate_interview_kit():
    st.title("📋 Generate Interview Kit")

    # Add debug info
    if st.session_state.debug:
        st.write(f"Backend URL: {BACKEND_URL}")
        st.write(f"Session state: {st.session_state}")

    with st.form("jd_form"):
        jd_text = st.text_area(
            "Paste Job Description",
            height=300,
            placeholder="Enter the complete job description here...\n\nExample:\nSenior Python Developer\nRequirements:\n- 5+ years Python experience\n- Django/FastAPI framework knowledge\n- AWS cloud experience\n- CI/CD pipelines\n\nResponsibilities:\n- Design and develop backend systems\n- Implement scalable solutions\n- Collaborate with frontend team"
        )

        submitted = st.form_submit_button("Generate Interview Kit", type="primary", width="stretch")

        if submitted and jd_text:
            if st.session_state.debug:
                st.write(f"Sending to backend: {jd_text[:100]}...")

            with st.spinner("Analyzing JD and generating interview questions..."):
                result = call_backend("/api/generate-interview-kit", "POST", {"text": jd_text})

                if result:
                    # Validate all questions
                    validated_questions = []
                    for q in result.get("questions", []):
                        validated_questions.append(validate_question_structure(q))
                    result["questions"] = validated_questions

                    st.session_state.interview_kit = result
                    st.session_state.download_data = json.dumps(result, indent=2)
                    st.success("Interview kit generated successfully!")
                else:
                    st.error("Failed to generate interview kit. Please check if the backend is running.")

    # Display results and download button OUTSIDE the form
    if st.session_state.interview_kit:
        result = st.session_state.interview_kit

        # Display results
        st.subheader("Generated Interview Kit")

        # Requirements overview
        with st.expander("Job Requirements Analysis"):
            st.json(result.get("job_requirements", {}))

        # Questions by type
        st.subheader("Interview Questions")

        for i, question in enumerate(result.get("questions", [])):
            with st.expander(f"Q{i + 1}: {question.get('question_text', 'No question text')}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Question Details**")
                    st.write(f"**Type:** {question.get('question_type', 'N/A')}")
                    st.write(f"**Difficulty:** {question.get('difficulty', 'N/A')}")
                    st.write(f"**Tests:** {question.get('tests_requirement', 'No tests specified')}")

                    if "quality_metrics" in question:
                        st.write(
                            f"**Quality Score:** {question['quality_metrics'].get('overall_score', 'N/A'):.2f}/1.0")

                with col2:
                    st.markdown("**Answer Guidelines**")
                    if "answer_guidelines" in question:
                        guidelines = question["answer_guidelines"]

                        # Safely access key_points with fallback
                        key_points = guidelines.get("key_points", [])
                        if key_points:
                            st.write("**Key Points:**")
                            for point in key_points:
                                st.write(f"• {point}")
                        else:
                            st.write("**Key Points:** Not available")

                        # Similarly protect other accesses
                        model_answer = guidelines.get("model_answer", "Not available")
                        with st.expander("View Ideal Answer"):
                            st.write(model_answer)

                        scoring_rubric = guidelines.get("scoring_rubric", {})
                        with st.expander("View Scoring Rubric"):
                            if scoring_rubric:
                                for score, criteria in scoring_rubric.items():
                                    st.write(f"**{score}:** {criteria}")
                            else:
                                st.write("Scoring rubric not available")
                    else:
                        st.write("Answer guidelines not available")

        # Download option - Now outside the form
        if st.session_state.download_data:
            st.download_button(
                label="Download Interview Kit",
                data=st.session_state.download_data,
                file_name=f"interview_kit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch"
            )


def conduct_interview():
    st.title("🎤 Conduct Interview")

    if not st.session_state.interview_kit:
        st.warning("Please generate an interview kit first from the 'Generate Interview Kit' page.")
        return

    kit = st.session_state.interview_kit

    # Interview session setup
    st.subheader("Interview Session")

    col1, col2 = st.columns(2)

    with col1:
        candidate_name = st.text_input("Candidate Name")
        candidate_role = st.text_input("Applied Position", value=kit.get("job_requirements", {}).get("job_title", ""))

    with col2:
        interview_date = st.date_input("Interview Date")
        interviewers = st.text_input("Interviewers (comma-separated)")

    # Question navigation - MOVED OUTSIDE THE COLUMN BLOCKS
    st.subheader("Interview Questions")

    questions = kit.get("questions", [])
    total_questions = len(questions)

    if total_questions == 0:
        st.error("No questions available in the interview kit. Please generate a new kit.")
        return

    # Create question navigation
    question_index = st.session_state.current_question_index

    # Ensure question_index is within valid range
    if question_index >= total_questions:
        st.session_state.current_question_index = 0
        question_index = 0

    # Safely get the current question
    question = questions[question_index] if question_index < total_questions else None

    if question is None:
        st.error("Invalid question index. Resetting to first question.")
        st.session_state.current_question_index = 0
        question = questions[0] if questions else None
        if question is None:
            return

    # Navigation controls
    nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

    with nav_col1:
        if st.button("◀ Previous", width="stretch") and question_index > 0:
            st.session_state.current_question_index -= 1
            st.rerun()

    with nav_col2:
        st.progress((question_index + 1) / total_questions)
        st.caption(f"Question {question_index + 1} of {total_questions}")

    with nav_col3:
        if st.button("Next ▶", width="stretch") and question_index < total_questions - 1:
            st.session_state.current_question_index += 1
            st.rerun()

    # Current question display - with additional safety checks
    st.markdown(f"### {question.get('question_text', 'Question text not available')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Question Details**")
        st.write(f"**Type:** {question.get('question_type', 'N/A')}")
        st.write(f"**Difficulty:** {question.get('difficulty', 'N/A')}")
        st.write(f"**Tests:** {question.get('tests_requirement', 'No tests specified')}")

        if "quality_metrics" in question:
            quality = question['quality_metrics']
            st.write(f"**Quality Score:** {quality.get('overall_score', 'N/A'):.2f}/1.0")

    with col2:
        st.markdown("**Scoring Guidelines**")
        # Safely access answer_guidelines with fallback
        if "answer_guidelines" in question:
            guidelines = question["answer_guidelines"]

            # Safely access all guideline components
            model_answer = guidelines.get("model_answer", "Not available")
            with st.expander("View Ideal Answer"):
                st.write(model_answer)

            scoring_rubric = guidelines.get("scoring_rubric", {})
            with st.expander("View Scoring Rubric"):
                if scoring_rubric:
                    st.json(scoring_rubric)
                else:
                    st.write("Scoring rubric not available")

            # Safely access key_points with fallback
            key_points = guidelines.get("key_points", [])
            if key_points:
                st.write("**Key Points:**")
                for point in key_points:
                    st.write(f"• {point}")
        else:
            st.write("Answer guidelines not available")

    # Response recording
    st.subheader("Candidate Response")
    response = st.text_area("Enter candidate's response", height=150, key=f"response_{question_index}")

    # Evaluation
    if st.button("Evaluate Response", type="primary", width="stretch", key=f"eval_{question_index}") and response:
        with st.spinner("Evaluating response..."):
            # Safely get question_id with fallback
            question_id = question.get("question_id", f"q{question_index}")

            eval_data = {
                "question_id": question_id,
                "response": response
            }

            evaluation = call_backend("/api/evaluate-response", "POST", {
                "jd": {"text": json.dumps(kit.get("job_requirements", {}))},
                "response": eval_data
            })

            if evaluation:
                st.success("Evaluation complete!")

                # Display evaluation results with safety checks
                eval_result = evaluation.get("evaluation", {})

                if eval_result:
                    col1, col2 = st.columns(2)

                    with col1:
                        overall_score = eval_result.get('overall_score', 0)
                        st.metric("Overall Score", f"{overall_score:.2f}/1.0")

                        # Score breakdown
                        st.markdown("**Score Breakdown**")
                        for metric, score in eval_result.items():
                            if metric != "overall_score":
                                st.write(f"{metric.replace('_', ' ').title()}: {score:.2f}")

                    with col2:
                        # Visualization with safety check
                        metrics = {k: v for k, v in eval_result.items() if
                                   k != "overall_score" and isinstance(v, (int, float))}
                        if metrics:
                            fig = px.bar(
                                x=list(metrics.keys()),
                                y=list(metrics.values()),
                                title="Evaluation Metrics",
                                labels={"x": "Metric", "y": "Score"}
                            )
                            st.plotly_chart(fig, width="stretch")
                        else:
                            st.write("No evaluation metrics available")
                else:
                    st.error("No evaluation results returned")
            else:
                st.error("Failed to evaluate response. Please check if the backend is running.")


def evaluate_responses():
    st.title("📊 Evaluate Responses")

    st.info("This feature allows you to evaluate candidate responses against multiple questions.")

    # File upload for batch processing
    uploaded_file = st.file_uploader("Upload responses JSON file", type="json")

    if uploaded_file:
        try:
            responses_data = json.load(uploaded_file)
            st.success("File uploaded successfully!")

            if "responses" in responses_data and "jd" in responses_data:
                # Display responses
                st.subheader("Candidate Responses")

                for i, response in enumerate(responses_data["responses"]):
                    with st.expander(f"Response {i + 1}: {response.get('question_id', 'Unknown')}"):
                        st.write("**Question:**", response.get("question_text", "Unknown"))
                        st.write("**Candidate Response:**", response.get("response", ""))

                        if st.button(f"Evaluate Response {i + 1}", key=f"eval_{i}", width="stretch"):
                            with st.spinner("Evaluating..."):
                                evaluation = call_backend("/api/evaluate-response", "POST", {
                                    "jd": responses_data["jd"],
                                    "response": response
                                })

                                if evaluation:
                                    st.json(evaluation)

        except json.JSONDecodeError:
            st.error("Invalid JSON file. Please upload a valid JSON file.")


def show_analytics():
    st.title("📈 Analytics & Reports")

    # Fetch analytics data from backend
    analytics_data = call_backend("/api/analytics/overview")

    if not analytics_data:
        # Sample data for demonstration
        analytics_data = {
            "summary_metrics": {
                "total_kits_generated": 24,
                "average_questions_per_kit": 12.5,
                "average_categories_per_jd": 6.2,
                "kits_last_7_days": 5
            },
            "time_analysis": {
                "daily_average": 3.4,
                "daily_std": 1.2,
                "busiest_day": "2023-10-15",
                "busiest_day_count": 7
            }
        }

    # Summary metrics
    st.subheader("Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Kits", analytics_data["summary_metrics"]["total_kits_generated"])
    with col2:
        st.metric("Avg Questions/Kit", f"{analytics_data['summary_metrics']['average_questions_per_kit']:.1f}")
    with col3:
        st.metric("Avg Categories/JD", f"{analytics_data['summary_metrics']['average_categories_per_jd']:.1f}")
    with col4:
        st.metric("Recent Activity", analytics_data["summary_metrics"]["kits_last_7_days"], "last 7 days")

    # Charts and visualizations
    st.subheader("Performance Trends")

    # Sample data for charts
    trend_data = pd.DataFrame({
        "Date": pd.date_range(start="2023-10-01", end="2023-10-20", freq="D"),
        "Kits_Generated": [2, 3, 1, 4, 2, 5, 3, 2, 4, 1, 3, 5, 2, 4, 3, 2, 1, 4, 3, 2],
        "Avg_Score": [7.2, 7.5, 7.8, 7.6, 7.9, 8.1, 8.2, 8.0, 7.8, 7.9, 8.1, 8.3, 8.2, 8.4, 8.1, 8.3, 8.0, 8.2, 8.4,
                      8.1]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.line(trend_data, x="Date", y="Kits_Generated",
                       title="Daily Interview Kits Generated")
        st.plotly_chart(fig1, width="stretch")

    with col2:
        fig2 = px.line(trend_data, x="Date", y="Avg_Score",
                       title="Average Candidate Scores Over Time")
        st.plotly_chart(fig2, width="stretch")

    # Question effectiveness analysis
    st.subheader("Question Effectiveness")

    # Sample question data
    question_data = pd.DataFrame({
        "Question_Type": ["Technical", "Behavioral", "Situational", "Cultural"],
        "Avg_Score": [7.8, 8.2, 7.5, 8.4],
        "Usage_Count": [45, 38, 28, 22],
        "Effectiveness": [0.82, 0.88, 0.76, 0.91]
    })

    col1, col2 = st.columns(2)

    with col1:
        fig3 = px.bar(question_data, x="Question_Type", y="Avg_Score",
                      title="Average Score by Question Type")
        st.plotly_chart(fig3, width="stretch")

    with col2:
        fig4 = px.scatter(question_data, x="Usage_Count", y="Effectiveness",
                          size="Avg_Score", color="Question_Type",
                          title="Question Effectiveness vs Usage")
        st.plotly_chart(fig4, width="stretch")

    # Export options
    st.subheader("Data Export")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Download Analytics Report (PDF)",
            data="Sample PDF content would be here",
            file_name="interview_analytics_report.pdf",
            mime="application/pdf",
            width="stretch"
        )

    with col2:
        st.download_button(
            "Export Raw Data (CSV)",
            data=trend_data.to_csv(index=False),
            file_name="interview_analytics_data.csv",
            mime="text/csv",
            width="stretch"
        )


if __name__ == "__main__":
    main()