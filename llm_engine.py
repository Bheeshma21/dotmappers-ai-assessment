import json
import os

from dotenv import load_dotenv
from groq import Groq

from query_engine import QueryEngine


load_dotenv()


class LLMEngine:
    """
    Handles the complete natural-language analytics pipeline.

    1. QueryEngine converts the question into a safe query plan.
    2. Pandas executes the query against the real CSV.
    3. The LLM converts the calculated result into a clear answer.
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found. "
                "Please add it to the .env file."
            )

        self.client = Groq(
            api_key=api_key
        )

        self.model = "openai/gpt-oss-20b"

        self.query_engine = QueryEngine()

    def generate_answer(
        self,
        question,
        query_plan,
        result,
    ):
        """
        Generate a business-friendly answer using only
        the Pandas result.
        """

        system_prompt = """
You are an AI support-ticket analytics assistant.

The user's question has already been analyzed and executed
against the real support-ticket CSV using Python and Pandas.

You will receive:
1. The original user question.
2. The structured query plan.
3. The actual Pandas result.

Rules:

1. Use ONLY the provided Pandas result.
2. Never invent numbers, tickets, agents, dates, or statistics.
3. Do not change calculated values.
4. Give a direct and concise answer.
5. If tickets are returned, summarize them clearly.
6. When useful, use a Markdown table for multiple tickets.
7. Do not mention internal implementation details unless asked.
8. Do not say you calculated a value if it was not present
   in the provided result.
"""

        user_prompt = f"""
USER QUESTION:
{question}

QUERY PLAN:
{json.dumps(query_plan, indent=2)}

ACTUAL PANDAS RESULT:
{json.dumps(result, indent=2)}

Answer the user's question using only the actual result above.
"""

        try:
            response = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    temperature=0.1,
                    max_tokens=800,
                )
            )

            return (
                response
                .choices[0]
                .message
                .content
                .strip()
            )

        except Exception as error:
            return (
                f"LLM answer generation failed: "
                f"{str(error)}"
            )

    def ask(self, question):
        """
        Complete natural-language analytics pipeline.
        """

        try:
            query_data = (
                self.query_engine.query(
                    question
                )
            )

            answer = self.generate_answer(
                question=question,
                query_plan=query_data[
                    "query_plan"
                ],
                result=query_data[
                    "result"
                ],
            )

            return answer

        except Exception as error:
            return (
                "I could not analyze that question. "
                f"Reason: {str(error)}"
            )


if __name__ == "__main__":
    llm = LLMEngine()

    print("=" * 60)
    print(
        "DOTMAPPERS DYNAMIC AI "
        "SUPPORT TICKET ASSISTANT"
    )
    print("=" * 60)

    question = (
        "Show me all Critical tickets "
        "with resolution time greater than 12 hours."
    )

    print("\nQuestion:")
    print(question)

    print("\nAI Answer:")
    print(
        llm.ask(question)
    )