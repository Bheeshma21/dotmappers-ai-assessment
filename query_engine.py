import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from data_engine import TicketDataEngine


load_dotenv()


class QueryEngine:
    """
    Converts a natural-language question into a safe,
    structured query plan and executes it with Pandas.

    The LLM never executes Python code.
    """

    ALLOWED_COLUMNS = [
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "agent_id",
        "customer_rating",
        "issue_summary",
    ]

    ALLOWED_OPERATIONS = [
        "count",
        "average",
        "sum",
        "min",
        "max",
        "list",
    ]

    ALLOWED_FILTER_OPERATORS = [
        "eq",
        "neq",
        "gt",
        "gte",
        "lt",
        "lte",
    ]

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY was not found in .env"
            )

        self.client = Groq(api_key=api_key)
        self.model = "openai/gpt-oss-20b"

        self.data_engine = TicketDataEngine()
        self.df = self.data_engine.get_dataframe()

    def create_query_plan(self, question):
        """
        Ask the LLM to translate natural language into JSON.
        """

        prompt = f"""
You convert support-ticket questions into structured JSON.

Available columns:

{", ".join(self.ALLOWED_COLUMNS)}

Valid category values:
Billing, General, Technical

Valid priority values:
Critical, High, Medium, Low

Valid status values:
Resolved, Open, Escalated

Valid operations:
count, average, sum, min, max, list

Valid filter operators:
eq, neq, gt, gte, lt, lte

Return ONLY valid JSON.

Do not return markdown.
Do not return Python.
Do not explain anything.

JSON format:

{{
    "operation": "count",
    "target_column": null,
    "filters": [
        {{
            "column": "priority",
            "operator": "eq",
            "value": "Critical"
        }}
    ],
    "group_by": null,
    "sort": "desc",
    "limit": 10
}}

Rules:

1. For "how many", use count.
2. For "average", use average.
3. For questions asking to show/find/list tickets, use list.
4. For "which agent/category/priority/status", use group_by.
5. For average resolution time, target_column must be
   resolution_time_hrs.
6. For average response time, target_column must be
   response_time_hrs.
7. For average rating, target_column must be customer_rating.
8. If the user asks about resolved tickets, status should be Resolved.
9. If the user asks about open tickets, status should be Open.
10. If the user asks about escalated tickets, status should be Escalated.
11. Never create a column that is not in the available columns.

Question:
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return only a valid JSON query plan."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=500,
            response_format={
                "type": "json_object"
            },
        )

        content = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

        return json.loads(content)

    def _validate_plan(self, plan):
        operation = plan.get("operation")

        if operation not in self.ALLOWED_OPERATIONS:
            raise ValueError(
                f"Unsupported operation: {operation}"
            )

        target = plan.get("target_column")

        if (
            target is not None
            and target not in self.ALLOWED_COLUMNS
        ):
            raise ValueError(
                f"Invalid target column: {target}"
            )

        group_by = plan.get("group_by")

        if (
            group_by is not None
            and group_by not in self.ALLOWED_COLUMNS
        ):
            raise ValueError(
                f"Invalid group_by column: {group_by}"
            )

        for filter_item in plan.get("filters", []):
            column = filter_item.get("column")
            operator = filter_item.get("operator")

            if column not in self.ALLOWED_COLUMNS:
                raise ValueError(
                    f"Invalid filter column: {column}"
                )

            if operator not in self.ALLOWED_FILTER_OPERATORS:
                raise ValueError(
                    f"Invalid filter operator: {operator}"
                )

    def _apply_filters(self, df, filters):
        result = df.copy()

        for item in filters:
            column = item["column"]
            operator = item["operator"]
            value = item["value"]

            series = result[column]

            if pd.api.types.is_numeric_dtype(series):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    pass

            if operator == "eq":
                result = result[series == value]

            elif operator == "neq":
                result = result[series != value]

            elif operator == "gt":
                result = result[series > value]

            elif operator == "gte":
                result = result[series >= value]

            elif operator == "lt":
                result = result[series < value]

            elif operator == "lte":
                result = result[series <= value]

        return result

    def execute_plan(self, plan):
        """
        Safely execute the structured plan using Pandas.
        """

        self._validate_plan(plan)

        df = self._apply_filters(
            self.df,
            plan.get("filters", []),
        )

        operation = plan["operation"]
        target = plan.get("target_column")
        group_by = plan.get("group_by")

        limit = plan.get("limit", 10)

        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 10

        limit = max(1, min(limit, 100))

        ascending = (
            plan.get("sort", "desc") == "asc"
        )

        if group_by:
            if operation == "count":
                result = (
                    df.groupby(group_by)
                    .size()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "average":
                if not target:
                    raise ValueError(
                        "Average requires target_column."
                    )

                result = (
                    df.groupby(group_by)[target]
                    .mean()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "sum":
                if not target:
                    raise ValueError(
                        "Sum requires target_column."
                    )

                result = (
                    df.groupby(group_by)[target]
                    .sum()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "min":
                if not target:
                    raise ValueError(
                        "Min requires target_column."
                    )

                result = (
                    df.groupby(group_by)[target]
                    .min()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "max":
                if not target:
                    raise ValueError(
                        "Max requires target_column."
                    )

                result = (
                    df.groupby(group_by)[target]
                    .max()
                    .sort_values(
                        ascending=ascending
                    )
                )

            else:
                raise ValueError(
                    "List operation cannot use group_by."
                )

            return result.head(limit).to_dict()

        if operation == "count":
            return {
                "count": int(len(df))
            }

        if operation == "average":
            if not target:
                raise ValueError(
                    "Average requires target_column."
                )

            return {
                "average": round(
                    float(df[target].mean()),
                    2,
                )
            }

        if operation == "sum":
            if not target:
                raise ValueError(
                    "Sum requires target_column."
                )

            return {
                "sum": round(
                    float(df[target].sum()),
                    2,
                )
            }

        if operation == "min":
            if not target:
                raise ValueError(
                    "Min requires target_column."
                )

            return {
                "min": round(
                    float(df[target].min()),
                    2,
                )
            }

        if operation == "max":
            if not target:
                raise ValueError(
                    "Max requires target_column."
                )

            return {
                "max": round(
                    float(df[target].max()),
                    2,
                )
            }

        if operation == "list":
            columns = [
                "ticket_id",
                "created_at",
                "category",
                "priority",
                "status",
                "response_time_hrs",
                "resolution_time_hrs",
                "agent_id",
                "customer_rating",
                "issue_summary",
            ]

            result = df[columns].head(limit).copy()

            result["created_at"] = (
                result["created_at"]
                .astype(str)
            )

            result = result.where(
                pd.notna(result),
                None,
            )

            return {
                "total_matches": int(len(df)),
                "tickets": result.to_dict(
                    orient="records"
                ),
            }

        raise ValueError(
            f"Unable to execute operation: {operation}"
        )

    def query(self, question):
        plan = self.create_query_plan(question)

        result = self.execute_plan(plan)

        return {
            "question": question,
            "query_plan": plan,
            "result": result,
        }


if __name__ == "__main__":
    engine = QueryEngine()

    question = (
        "Show me all Critical tickets "
        "with resolution time greater than 12 hours."
    )
    

    print("=" * 60)
    print("DOTMAPPERS NATURAL LANGUAGE QUERY ENGINE")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    result = engine.query(question)

    print("\nGenerated Query Plan:")
    print(
        json.dumps(
            result["query_plan"],
            indent=2,
        )
    )

    print("\nPandas Result:")
    print(
        json.dumps(
            result["result"],
            indent=2,
        )
    )