# Enterprise JD-Centric Interview System

A comprehensive AI-powered interview system that generates questions and evaluation criteria based strictly on job descriptions.

## Features

- JD-based question generation
- Advanced candidate response evaluation
- Real-time collaboration for interview teams
- Comprehensive analytics and reporting
- Quality assurance for interview questions

## Setup

1. Ensure Docker and Docker Compose are installed
2. Clone this repository
3. Run: `docker-compose up`
4. The API will be available at http://localhost:8000

## API Endpoints

- `POST /api/generate-interview-kit` - Generate interview kit from JD
- `POST /api/evaluate-response` - Evaluate candidate responses
- `POST /api/create-interview-session` - Create collaborative session
- `POST /api/submit-feedback` - Submit interview feedback
- `GET /api/analytics/overview` - Get analytics overview
- `GET /api/health` - Health check

## Usage

1. Start Ollama and load models: `ollama pull llama3:70b`, `ollama pull mixtral`
2. Start the application: `docker-compose up`
3. Use the API to generate interview kits and evaluate responses

## Models Required

- llama3:70b
- mixtral
- all-MiniLM-L6-v2 (automatically downloaded by sentence-transformers)