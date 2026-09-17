import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from anomaly_detector import AnomalyDetector
from query_engine import QueryEngine


load_dotenv()


class LLMEngine:
    """
    Handles the complete natural-language analytics pipeline.

    Normal analytics questions:
    1. QueryEngine converts the question into a safe query plan.
    2. Pandas executes the query against the real CSV.
    3. The LLM converts the calculated result into a clear answer.

    Anomaly questions:
    1. Detect anomaly intent.
    2. Apply the requested date scope.
    3. Run deterministic anomaly detection.
    4. Return the real anomaly results to the LLM for formatting.
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

    def _is_anomaly_question(self, question):
        """
        Detect whether the user is asking about anomalies
        or abnormal/outlier ticket behavior.
        """
        question_lower = question.lower()

        anomaly_terms = [
            "anomaly",
            "anomalies",
            "anomalous",
            "abnormal",
            "abnormally",
            "outlier",
            "outliers",
        ]

        return any(
            term in question_lower
            for term in anomaly_terms
        )

    def _get_date_scoped_dataframe(
        self,
        question,
    ):
        """
        Apply simple deterministic date scopes for
        anomaly questions.

        Historical relative dates are calculated using
        the latest timestamp in the dataset.
        """
        df = self.query_engine.df.copy()

        if df.empty:
            return df, "all data"

        question_lower = question.lower()

        reference_time = (
            pd.to_datetime(
                df["created_at"]
            ).max()
        )

        if "this week" in question_lower:
            start = (
                reference_time.normalize()
                - pd.Timedelta(
                    days=reference_time.weekday()
                )
            )

            scoped_df = df[
                (df["created_at"] >= start)
                & (
                    df["created_at"]
                    <= reference_time
                )
            ].copy()

            date_scope = (
                f"{start.strftime('%Y-%m-%d')} "
                f"to "
                f"{reference_time.strftime('%Y-%m-%d')}"
            )

            return scoped_df, date_scope

        if "this month" in question_lower:
            start = reference_time.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            scoped_df = df[
                (df["created_at"] >= start)
                & (
                    df["created_at"]
                    <= reference_time
                )
            ].copy()

            date_scope = (
                f"{start.strftime('%Y-%m-%d')} "
                f"to "
                f"{reference_time.strftime('%Y-%m-%d')}"
            )

            return scoped_df, date_scope

        return df, "all data"

    def _handle_anomaly_question(
        self,
        question,
    ):
        """
        Run deterministic anomaly detection for
        natural-language anomaly questions.
        """
        scoped_df, date_scope = (
            self._get_date_scoped_dataframe(
                question
            )
        )

        if scoped_df.empty:
            return {
                "type": "anomaly_analysis",
                "date_scope": date_scope,
                "total_tickets_checked": 0,
                "anomaly_count": 0,
                "anomalies": [],
            }

        detector = AnomalyDetector(
            scoped_df
        )

        detected = (
            detector.detect_anomalies()
        )

        question_lower = question.lower()

        # If the question specifically asks about
        # resolution-time anomalies, return only those.
        if (
            "resolution" in question_lower
            and "time" in question_lower
        ):
            threshold = (
                detector
                .get_thresholds()[
                    "resolution_time_threshold"
                ]
            )

            anomalies = detected[
                detected[
                    "resolution_time_hrs"
                ].notna()
                & (
                    detected[
                        "resolution_time_hrs"
                    ] > threshold
                )
            ].copy()

            columns = [
                "ticket_id",
                "created_at",
                "category",
                "priority",
                "status",
                "resolution_time_hrs",
                "agent_id",
                "customer_rating",
                "anomaly_reasons",
            ]

            records = (
                anomalies[columns]
                .astype(object)
                .where(
                    anomalies[columns].notna(),
                    None,
                )
                .to_dict(
                    orient="records"
                )
            )

            # Make timestamps JSON-safe.
            for record in records:
                created_at = record.get(
                    "created_at"
                )

                if isinstance(
                    created_at,
                    pd.Timestamp,
                ):
                    record["created_at"] = (
                        created_at.isoformat()
                    )

            return {
                "type": (
                    "resolution_time_anomalies"
                ),
                "date_scope": date_scope,
                "total_tickets_checked": int(
                    len(scoped_df)
                ),
                "resolution_time_threshold_hrs": (
                    float(threshold)
                ),
                "anomaly_count": int(
                    len(anomalies)
                ),
                "anomalies": records,
            }

        # General anomaly question.
        columns = [
            "ticket_id",
            "created_at",
            "category",
            "priority",
            "status",
            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating",
            "anomaly_reasons",
        ]

        records = (
            detected[columns]
            .astype(object)
            .where(
                detected[columns].notna(),
                None,
            )
            .to_dict(
                orient="records"
            )
        )

        for record in records:
            created_at = record.get(
                "created_at"
            )

            if isinstance(
                created_at,
                pd.Timestamp,
            ):
                record["created_at"] = (
                    created_at.isoformat()
                )

        return {
            "type": "anomaly_analysis",
            "date_scope": date_scope,
            "total_tickets_checked": int(
                len(scoped_df)
            ),
            "anomaly_count": int(
                len(detected)
            ),
            "anomalies": records,
        }

    def generate_answer(
        self,
        question,
        query_plan,
        result,
    ):
        """
        Generate a business-friendly answer using only
        the deterministic Pandas result.
        """

        system_prompt = """
You are an AI support-ticket analytics assistant.

The user's question has already been analyzed and executed
against the real support-ticket CSV using Python and Pandas.

You will receive:
1. The original user question.
2. The structured query plan or analysis type.
3. The actual deterministic result.

Rules:

1. Use ONLY the provided result.
2. Never invent numbers, tickets, agents, dates, or statistics.
3. Do not change calculated values.
4. Give a direct and concise answer.
5. If tickets are returned, summarize them clearly.
6. When useful, use a Markdown table for multiple tickets.
7. Do not mention internal implementation details unless asked.
8. Do not say you calculated a value if it was not present
   in the provided result.
9. For anomaly questions, clearly state whether anomalies
   were found and how many were found.
10. If an anomaly threshold is provided, include it when useful.
"""

        user_prompt = f"""
USER QUESTION:
{question}

QUERY PLAN / ANALYSIS TYPE:
{json.dumps(query_plan, indent=2)}

ACTUAL RESULT:
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
                    max_tokens=3000,
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
            if self._is_anomaly_question(
                question
            ):
                result = (
                    self._handle_anomaly_question(
                        question
                    )
                )

                query_plan = {
                    "operation": (
                        "anomaly_detection"
                    ),
                    "date_scope": result.get(
                        "date_scope"
                    ),
                    "analysis_type": result.get(
                        "type"
                    ),
                }

                return self.generate_answer(
                    question=question,
                    query_plan=query_plan,
                    result=result,
                )

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
        "Are there any anomalies in "
        "resolution times this week?"
    )

    print("\nQuestion:")
    print(question)

    print("\nAI Answer:")
    print(
        llm.ask(question)
    )