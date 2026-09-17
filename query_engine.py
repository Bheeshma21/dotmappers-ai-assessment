import json
import os

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from data_engine import TicketDataEngine


load_dotenv()


class QueryEngine:
    """
    Converts natural-language support-ticket questions
    into validated structured query plans and executes
    them safely with Pandas.

    The LLM never executes Python code.

    Relative dates use the latest timestamp in the
    historical dataset as the reference time.
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

    ALLOWED_DATE_RANGES = [
        "this_week",
        "this_month",
        None,
    ]

    ALLOWED_SPECIAL_CONDITIONS = [
        "resolution_sla_breach_hours",
        None,
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

    def _reference_time(self):
        """
        Return the latest created_at timestamp.

        Because this is a historical assessment dataset,
        relative periods and unresolved-ticket ages use
        this timestamp rather than the computer clock.
        """

        reference_time = self.df["created_at"].max()

        if pd.isna(reference_time):
            raise ValueError(
                "Dataset contains no valid created_at values."
            )

        return reference_time

    def create_query_plan(self, question):
        """
        Ask Groq to translate a natural-language question
        into a safe JSON query plan.

        Structured-output failures are retried up to
        three times.
        """

        reference_time = self._reference_time()

        prompt = f"""
You convert support-ticket questions into structured JSON.

The dataset is historical.

Latest timestamp in the dataset:
{reference_time}

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

Valid date_range values:
null, this_week, this_month

Valid special_condition values:
null, resolution_sla_breach_hours

Return ONLY one valid JSON object.

Do not return markdown.
Do not return Python.
Do not explain the result.
Do not create unsupported fields.

JSON format:

{{
    "operation": "count",
    "target_column": null,
    "filters": [],
    "group_by": null,
    "sort": "desc",
    "limit": 10,
    "date_range": null,
    "special_condition": null,
    "special_condition_value": null
}}

Rules:

1. For "how many", use count.

2. For "average", use average.

3. For questions asking to show, find, or list
   tickets, use list.

4. For questions asking "which agent",
   "which category", "which priority", or
   "which status", use group_by.

5. For average resolution time:
   target_column = resolution_time_hrs.

6. For average response time:
   target_column = response_time_hrs.

7. For average customer rating:
   target_column = customer_rating.

8. If the user specifically asks about tickets
   that ARE resolved, add:
   status eq Resolved.

9. If the user asks about open tickets, add:
   status eq Open.

10. If the user asks about escalated tickets, add:
    status eq Escalated.

11. If the question says "this month":
    date_range = this_month.

12. If the question says "this week":
    date_range = this_week.

13. Otherwise:
    date_range = null.

14. "Most" or "highest" normally means:
    sort = desc.

15. "Least" or "lowest" normally means:
    sort = asc.

16. Never invent columns.

17. Never invent category, priority, or
    status values.

18. IMPORTANT SLA RULE:

    If the user says:

    "not resolved within X hours"

    use:

    "special_condition":
        "resolution_sla_breach_hours"

    "special_condition_value": X

    Do NOT add a resolution_time_hrs filter
    for this phrase.

    Do NOT add status eq Resolved.

19. resolution_sla_breach_hours means:

    A ticket matches when EITHER:

    A) it was resolved but
       resolution_time_hrs > X

    OR

    B) resolution_time_hrs is null
       and the ticket has been open for
       more than X hours relative to the
       latest dataset timestamp.

20. Example:

Question:
Show me all Critical tickets not resolved within 12 hours.

Correct JSON:

{{
    "operation": "list",
    "target_column": null,
    "filters": [
        {{
            "column": "priority",
            "operator": "eq",
            "value": "Critical"
        }}
    ],
    "group_by": null,
    "sort": null,
    "limit": 100,
    "date_range": null,
    "special_condition":
        "resolution_sla_breach_hours",
    "special_condition_value": 12
}}

21. For a list request containing the word "all",
    use limit 100.

22. If no special condition is required:

    "special_condition": null
    "special_condition_value": null

Question:
{question}
"""

        last_error = None

        for attempt in range(3):
            try:
                response = (
                    self.client
                    .chat
                    .completions
                    .create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a JSON query "
                                    "planner. Return exactly "
                                    "one valid JSON object "
                                    "matching the requested "
                                    "schema. Do not include "
                                    "explanations."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        temperature=0,
                        max_tokens=1300,
                        response_format={
                            "type": "json_object"
                        },
                    )
                )

                content = (
                    response
                    .choices[0]
                    .message
                    .content
                    .strip()
                )

                plan = json.loads(content)

                return plan

            except Exception as error:
                last_error = error

                if attempt == 2:
                    raise RuntimeError(
                        "The LLM could not generate "
                        "a valid query plan after "
                        "3 attempts."
                    ) from last_error

        raise RuntimeError(
            "Unable to create query plan."
        )

    def _validate_plan(self, plan):
        """
        Validate all LLM-generated fields before
        executing anything against the dataframe.
        """

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

        date_range = plan.get("date_range")

        if date_range not in self.ALLOWED_DATE_RANGES:
            raise ValueError(
                f"Invalid date_range: {date_range}"
            )

        special_condition = plan.get(
            "special_condition"
        )

        if (
            special_condition
            not in self.ALLOWED_SPECIAL_CONDITIONS
        ):
            raise ValueError(
                "Invalid special_condition: "
                f"{special_condition}"
            )

        filters = plan.get(
            "filters",
            [],
        )

        if not isinstance(filters, list):
            raise ValueError(
                "filters must be a list."
            )

        for filter_item in filters:
            column = filter_item.get(
                "column"
            )

            operator = filter_item.get(
                "operator"
            )

            if column not in self.ALLOWED_COLUMNS:
                raise ValueError(
                    f"Invalid filter column: {column}"
                )

            if (
                operator
                not in self.ALLOWED_FILTER_OPERATORS
            ):
                raise ValueError(
                    "Invalid filter operator: "
                    f"{operator}"
                )

            if "value" not in filter_item:
                raise ValueError(
                    "Filter is missing value."
                )

        if (
            special_condition
            == "resolution_sla_breach_hours"
        ):
            value = plan.get(
                "special_condition_value"
            )

            try:
                value = float(value)

            except (TypeError, ValueError):
                raise ValueError(
                    "resolution_sla_breach_hours "
                    "requires a numeric value."
                )

            if value < 0:
                raise ValueError(
                    "SLA hours cannot be negative."
                )

    def _apply_date_range(
        self,
        df,
        date_range,
    ):
        """
        Apply relative date filters using the latest
        dataset timestamp.
        """

        if not date_range:
            return df

        reference_time = self._reference_time()

        if date_range == "this_month":
            start = reference_time.replace(
                day=1,
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )

            return df[
                (df["created_at"] >= start)
                & (
                    df["created_at"]
                    <= reference_time
                )
            ]

        if date_range == "this_week":
            start_of_day = (
                reference_time.normalize()
            )

            start = (
                start_of_day
                - pd.Timedelta(
                    days=reference_time.weekday()
                )
            )

            return df[
                (df["created_at"] >= start)
                & (
                    df["created_at"]
                    <= reference_time
                )
            ]

        return df

    def _apply_filters(
        self,
        df,
        filters,
    ):
        """
        Apply ordinary AND filters from the
        validated query plan.
        """

        result = df.copy()

        for item in filters:
            column = item["column"]
            operator = item["operator"]
            value = item["value"]

            series = result[column]

            if pd.api.types.is_numeric_dtype(
                series
            ):
                try:
                    value = float(value)

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if operator == "eq":
                result = result[
                    series == value
                ]

            elif operator == "neq":
                result = result[
                    series != value
                ]

            elif operator == "gt":
                result = result[
                    series > value
                ]

            elif operator == "gte":
                result = result[
                    series >= value
                ]

            elif operator == "lt":
                result = result[
                    series < value
                ]

            elif operator == "lte":
                result = result[
                    series <= value
                ]

        return result

    def _apply_special_condition(
        self,
        df,
        condition,
        value,
    ):
        """
        Apply validated business conditions that
        require logic beyond a simple AND filter.
        """

        if not condition:
            return df

        if (
            condition
            == "resolution_sla_breach_hours"
        ):
            hours = float(value)

            reference_time = self._reference_time()

            ticket_age_hrs = (
                (
                    reference_time
                    - df["created_at"]
                )
                .dt
                .total_seconds()
                / 3600
            )

            resolved_late = (
                df["resolution_time_hrs"].notna()
                & (
                    df["resolution_time_hrs"]
                    > hours
                )
            )

            still_unresolved_late = (
                df["resolution_time_hrs"].isna()
                & (
                    ticket_age_hrs
                    > hours
                )
            )

            return df[
                resolved_late
                | still_unresolved_late
            ]

        return df

    @staticmethod
    def _safe_number(value):
        """
        Convert Pandas numeric results into
        JSON-safe values.
        """

        if pd.isna(value):
            return None

        return round(
            float(value),
            2,
        )

    def execute_plan(self, plan):
        """
        Execute a validated structured plan
        using Pandas.
        """

        self._validate_plan(plan)

        df = self._apply_date_range(
            self.df,
            plan.get("date_range"),
        )

        df = self._apply_filters(
            df,
            plan.get(
                "filters",
                [],
            ),
        )

        df = self._apply_special_condition(
            df,
            plan.get(
                "special_condition"
            ),
            plan.get(
                "special_condition_value"
            ),
        )

        operation = plan["operation"]

        target = plan.get(
            "target_column"
        )

        group_by = plan.get(
            "group_by"
        )

        limit = plan.get(
            "limit",
            10,
        )

        try:
            limit = int(limit)

        except (
            TypeError,
            ValueError,
        ):
            limit = 10

        limit = max(
            1,
            min(
                limit,
                100,
            ),
        )

        ascending = (
            plan.get(
                "sort",
                "desc",
            )
            == "asc"
        )

        if group_by:
            if operation == "count":
                result = (
                    df
                    .groupby(group_by)
                    .size()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "average":
                if not target:
                    raise ValueError(
                        "Average requires "
                        "target_column."
                    )

                result = (
                    df
                    .groupby(group_by)[target]
                    .mean()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "sum":
                if not target:
                    raise ValueError(
                        "Sum requires "
                        "target_column."
                    )

                result = (
                    df
                    .groupby(group_by)[target]
                    .sum()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "min":
                if not target:
                    raise ValueError(
                        "Min requires "
                        "target_column."
                    )

                result = (
                    df
                    .groupby(group_by)[target]
                    .min()
                    .sort_values(
                        ascending=ascending
                    )
                )

            elif operation == "max":
                if not target:
                    raise ValueError(
                        "Max requires "
                        "target_column."
                    )

                result = (
                    df
                    .groupby(group_by)[target]
                    .max()
                    .sort_values(
                        ascending=ascending
                    )
                )

            else:
                raise ValueError(
                    "List operation cannot "
                    "use group_by."
                )

            result = result.head(limit)

            clean_result = {}

            for key, value in result.items():
                if operation == "count":
                    clean_result[
                        str(key)
                    ] = int(value)

                else:
                    clean_result[
                        str(key)
                    ] = (
                        self._safe_number(
                            value
                        )
                    )

            return clean_result

        if operation == "count":
            return {
                "count": int(len(df))
            }

        if operation == "average":
            if not target:
                raise ValueError(
                    "Average requires "
                    "target_column."
                )

            return {
                "average": (
                    self._safe_number(
                        df[target].mean()
                    )
                )
            }

        if operation == "sum":
            if not target:
                raise ValueError(
                    "Sum requires "
                    "target_column."
                )

            return {
                "sum": (
                    self._safe_number(
                        df[target].sum()
                    )
                )
            }

        if operation == "min":
            if not target:
                raise ValueError(
                    "Min requires "
                    "target_column."
                )

            return {
                "min": (
                    self._safe_number(
                        df[target].min()
                    )
                )
            }

        if operation == "max":
            if not target:
                raise ValueError(
                    "Max requires "
                    "target_column."
                )

            return {
                "max": (
                    self._safe_number(
                        df[target].max()
                    )
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

            result = (
                df[columns]
                .head(limit)
                .copy()
            )

            result["created_at"] = (
                result["created_at"]
                .astype(str)
            )

            result = (
                result
                .astype(object)
                .where(
                    pd.notna(result),
                    None,
                )
            )

            return {
                "total_matches": int(
                    len(df)
                ),
                "tickets": (
                    result.to_dict(
                        orient="records"
                    )
                ),
            }

        raise ValueError(
            "Unable to execute operation: "
            f"{operation}"
        )

    def query(self, question):
        plan = self.create_query_plan(
            question
        )

        result = self.execute_plan(
            plan
        )

        return {
            "question": question,
            "query_plan": plan,
            "result": result,
        }


if __name__ == "__main__":
    engine = QueryEngine()

    question = (
        "Show me all Critical tickets "
        "not resolved within 12 hours."
    )

    print("=" * 60)

    print(
        "DOTMAPPERS NATURAL LANGUAGE "
        "QUERY ENGINE"
    )

    print("=" * 60)

    print("\nReference Time:")
    print(engine._reference_time())

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