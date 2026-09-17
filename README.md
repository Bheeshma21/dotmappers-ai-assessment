# AI Support Ticket Analyst

An AI-powered support ticket analytics system built for the DotMappers AI Intern Assessment.

The application allows users to ask natural-language questions about support ticket data, performs safe data analysis using Pandas, detects anomalous tickets, exposes REST API endpoints using FastAPI, and provides an interactive Streamlit interface.

## Features

- Natural-language querying of support ticket data
- LLM-powered query interpretation using Groq
- Safe structured query execution using Pandas
- Date-aware queries such as "this week" and "this month"
- SLA-style queries for unresolved and late-resolved tickets
- Statistical and rule-based anomaly detection
- FastAPI REST API
- Streamlit web interface
- CSV data validation and preprocessing
- Automated tests with Pytest
- API key protection using environment variables
- Single-command application startup

## Architecture

```text
User
  |
  v
Streamlit UI
  |
  v
FastAPI REST API
  |
  v
LLM Engine
  |
  +--------------------------+
  |                          |
  v                          v
Query Engine             Anomaly Detector
  |
  v
Groq LLM
  |
  v
Validated JSON Query Plan
  |
  v
Pandas Data Analysis
  |
  v
support_tickets.csv
```

The LLM does not execute arbitrary Python code.

For normal analytical questions, the LLM converts the user's natural-language question into a restricted structured JSON query plan. The plan is validated against approved operations, columns, filters, and operators before Pandas executes it against the supplied CSV.

The calculated result is then passed back to the LLM to produce a clear natural-language answer.

Anomaly-related questions are routed to the deterministic anomaly detection logic so anomaly calculations come directly from the dataset rather than being invented by the LLM.

## Technology Stack

- Python
- Pandas
- FastAPI
- Uvicorn
- Streamlit
- Groq LLM API
- python-dotenv
- Requests
- Pytest

## Project Structure

```text
dotmappers-ai-assessment/
|
|-- data/
|   `-- support_tickets.csv
|
|-- tests/
|   |-- test_data_engine.py
|   |-- test_anomaly_detector.py
|   `-- test_query_engine.py
|
|-- app.py
|-- api.py
|-- anomaly_detector.py
|-- data_engine.py
|-- llm_engine.py
|-- query_engine.py
|-- run.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## Dataset

The supplied dataset contains 500 support tickets.

Main fields include:

- ticket_id
- created_at
- category
- priority
- status
- response_time_hrs
- resolution_time_hrs
- agent_id
- customer_rating
- issue_summary

The application validates required columns and converts date and numerical fields into appropriate data types before analysis.

For unresolved tickets, `resolution_time_hrs` can be empty. The system therefore calculates ticket age from `created_at` when evaluating unresolved-ticket SLA conditions.

## Natural-Language Querying

Example questions:

```text
How many tickets are currently open?
```

```text
Which agent resolved the most tickets this month?
```

```text
Show me all Critical tickets not resolved within 12 hours.
```

```text
What is the average customer rating for Technical category tickets?
```

```text
Are there any anomalies in resolution times this week?
```

For normal analytical questions, the LLM generates a restricted JSON query plan.

Example:

```json
{
  "operation": "count",
  "target_column": null,
  "filters": [
    {
      "column": "status",
      "operator": "eq",
      "value": "Open"
    }
  ],
  "group_by": null,
  "sort": "desc",
  "limit": 10
}
```

Pandas performs the actual filtering, grouping, counting, and calculations.

This design reduces hallucination risk because numerical results are calculated from the dataset rather than generated directly by the LLM.

## Supported Query Operations

The query engine supports:

- count
- average
- sum
- minimum
- maximum
- list/filter operations
- grouping
- sorting
- multiple filters
- date filters
- SLA-style resolution conditions

Supported comparison operators include:

- equal
- not equal
- greater than
- greater than or equal
- less than
- less than or equal

## Date-Aware Queries

The system supports relative date expressions such as:

```text
this week
```

and:

```text
this month
```

Because the supplied dataset is historical, relative dates are evaluated against the latest timestamp present in the dataset rather than the computer's current date.

For this dataset, the latest ticket timestamp is:

```text
2024-03-30 18:06:00
```

This makes historical queries deterministic and meaningful during evaluation.

## SLA Query Handling

Questions such as:

```text
Show me all Critical tickets not resolved within 12 hours.
```

require more than a simple `resolution_time_hrs > 12` filter.

The system handles both cases:

1. Tickets that were resolved but took longer than the requested SLA.
2. Tickets that are still unresolved and have already been open longer than the requested SLA.

For unresolved tickets, age is calculated from `created_at` relative to the dataset reference time.

## Anomaly Detection

The project combines statistical anomaly detection with business rules.

### Statistical Detection

The Interquartile Range (IQR) method is used to detect unusually high:

- response times
- resolution times

The upper threshold is calculated using:

```text
Q3 + 1.5 * IQR
```

For the complete supplied dataset:

```text
Response time threshold: 7.65 hours
Resolution time threshold: 48.15 hours
```

### Business Rules

Tickets are additionally flagged when:

- customer rating is 2 or lower
- High or Critical priority tickets remain unresolved for more than 24 hours

For unresolved tickets, ticket age is calculated relative to the latest timestamp in the supplied historical dataset.

### Current Anomaly Results

```text
Total tickets: 500
Detected anomalies: 145
Anomaly percentage: 29.0%
```

The anomaly detector records the reason each ticket was flagged.

## Example Evaluation Results

The supplied assessment example questions were tested successfully.

### 1. Open Tickets

Question:

```text
How many tickets are currently open?
```

Result:

```text
111 tickets
```

### 2. Agent With Most Resolutions This Month

Question:

```text
Which agent resolved the most tickets this month?
```

Result:

```text
AGT-01 — 16 resolved tickets
```

### 3. Critical Tickets Outside 12-Hour Resolution SLA

Question:

```text
Show me all Critical tickets not resolved within 12 hours.
```

Result:

```text
34 matching tickets
```

The result includes both late-resolved tickets and tickets that remain unresolved beyond 12 hours.

### 4. Technical Ticket Customer Rating

Question:

```text
What is the average customer rating for Technical category tickets?
```

Result:

```text
3.74 / 5
```

### 5. Resolution-Time Anomalies This Week

Question:

```text
Are there any anomalies in resolution times this week?
```

Result:

```text
2 resolution-time anomalies
TKT-108
TKT-130
```

For that week's data, the calculated IQR resolution-time threshold is 87.55 hours.

## REST API

The FastAPI backend provides the following endpoints.

### Health Check

```http
GET /health
```

Returns API status and the number of loaded tickets.

Example:

```json
{
  "status": "healthy",
  "total_tickets": 500
}
```

### Dataset Summary

```http
GET /summary
```

Returns overall support-ticket statistics.

### Ask AI

```http
POST /ask
```

Example request:

```json
{
  "question": "How many tickets are currently open?"
}
```

Example response:

```json
{
  "question": "How many tickets are currently open?",
  "answer": "There are 111 tickets currently open."
}
```

### Anomaly Detection

```http
GET /anomalies
```

Returns the anomaly summary and detected anomalous tickets.

### API Documentation

When the application is running, Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Streamlit Interface

The Streamlit interface provides:

- API connection status
- dataset overview
- natural-language question input
- AI-generated answers
- anomaly statistics
- expandable anomalous-ticket table

The local interface is available at:

```text
http://localhost:8501
```

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd dotmappers-ai-assessment
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key
```

A Groq API key is required for natural-language LLM queries.

The project is designed to use Groq's free-access API tier, so the evaluator does not need a paid API service.

The `.env` file is excluded from Git through `.gitignore`.

Never commit API keys to the repository.

## Running the Application

The complete application can be started with a single command:

```bash
python run.py
```

This starts:

```text
FastAPI backend: http://127.0.0.1:8000
Streamlit UI:    http://localhost:8501
```

The backend and frontend can also be started separately for development.

FastAPI:

```bash
uvicorn api:app --reload
```

Streamlit:

```bash
streamlit run app.py
```

## Testing

Run all automated tests with:

```bash
python -m pytest -v
```

Current test suite:

```text
22 passed
```

Tests cover:

- dataset loading
- required columns
- categories
- priorities
- statuses
- dataset summary
- analysis context
- anomaly count
- anomaly thresholds
- anomaly output structure
- anomaly reasons
- unresolved High/Critical tickets older than 24 hours
- historical reference time
- basic count queries
- average calculations
- grouped queries
- this-month filtering
- this-week filtering
- resolution SLA handling
- invalid query-column rejection

## Security and Reliability

The application follows several safety practices:

- API keys are loaded from environment variables
- `.env` is excluded from Git
- the LLM cannot execute arbitrary Python
- generated query plans are validated before execution
- only approved dataset columns can be queried
- only approved query operations and filter operators are executed
- deterministic Pandas calculations produce numerical results
- anomaly calculations use deterministic statistical and business rules
- invalid or unsupported query plans are rejected

## Design Decisions

### Why use an LLM?

The LLM is used to understand flexible natural-language questions and translate them into a structured query representation.

### Why not let the LLM analyze the CSV directly?

Allowing an LLM to generate numerical answers directly can introduce hallucinations. Instead, Pandas calculates the result from the actual dataset.

### Why use a validated query plan?

A restricted query schema provides more predictable and secure execution than running arbitrary LLM-generated Python code.

### Why use the latest dataset timestamp?

The dataset is historical. Using its latest timestamp makes queries such as "this week", "this month", and "older than 24 hours" meaningful and reproducible during evaluation.

## Limitations

- Natural-language interpretation depends on the LLM correctly mapping questions to the supported query-plan schema.
- The query engine intentionally supports a restricted set of analytical operations for safety.
- A valid Groq API key is required for LLM-powered natural-language querying.
- Free-tier API rate limits may apply.
- The implementation is designed specifically for the supplied support-ticket schema.
- Very complex analytical questions outside the supported query-plan operations may require additional query-engine capabilities.

## Author

Bheeshma Reddy

AI Intern Assessment — DotMappers IT Pvt. Ltd.