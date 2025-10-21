# KG-EntityResearcher - LangGraph Two Agent System

Perform research on entities, and build entity KG framework using research results.
A two-agent system using LangGraph where a Researcher agent performs web searches using Serper API and an Analyst agent churning final result.

## Features

- **Researcher Agent**: Performs real-time web searches using Serper API
- **Analyst Agent**: Reviews research and consolidate search results, or requests additional information
- **Iterative Process**: Agents collaborate until sufficient information is gathered
- **Docker Support**: Easy deployment with Docker and Docker Compose

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- OpenAI API Key
- Serper API Key (get free credits at [serper.dev](https://serper.dev))

## Setup

### Local Setup

1. Clone the repository:
```bash
git clone 
cd KG-EntityResearcher
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

5. Run the application:
```bash
python main.py --subject "实体例如:应用场景、产品功效" --industry "行业"
```

### Docker Setup

1. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

3. Curl Usage:
```bash
curl -X POST http://localhost:5000/research \
  -H "Content-Type: application/json" \
  -d '{"subject": "实体例如:应用场景、产品功效", "industry": "行业"}'
```

## Usage

Edit `main.py` to customize the research subject:

```python
if __name__ == "__main__":
    subject = "Your research subject here"
    industry = "Your research industry here"
    agent_system_init = run_research_report()
    result = agent_system_init.run(subject,industry)
```

## Architecture

```
┌─────────────┐
│   Subject   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Researcher  │──► Performs web search
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Analyst    │──► Reviews & decides
└──────┬──────┘
       │
       ├──► Needs more info? ──► Back to Researcher
       │
       └──► Sufficient? ──► Final Result
```
