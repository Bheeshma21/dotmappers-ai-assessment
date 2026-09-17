import pandas as pd

from query_engine import QueryEngine


def make_engine():
    """
    Create QueryEngine without calling the Groq API.
    These tests verify deterministic Pandas execution.
    """
    engine = QueryEngine.__new__(QueryEngine)

    engine.df = pd.DataFrame(
        {
            "ticket_id": [
                "TKT-001",
                "TKT-002",
                "TKT-003",
                "TKT-004",
                "TKT-005",
            ],
            "created_at": pd.to_datetime(
                [
                    "2024-03-29 20:00:00",
                    "2024-03-30 10:00:00",
                    "2024-03-20 10:00:00",
                    "2024-03-28 10:00:00",
                    "2024-02-15 10:00:00",
                ]
            ),
            "category": [
                "Technical",
                "Billing",
                "Technical",
                "General",
                "Billing",
            ],
            "priority": [
                "Critical",
                "Critical",
                "High",
                "Critical",
                "Low",
            ],
            "status": [
                "Open",
                "Resolved",
                "Resolved",
                "Resolved",
                "Open",
            ],
            "response_time_hrs": [
                2.0,
                3.0,
                1.0,
                2.0,
                4.0,
            ],
            "resolution_time_hrs": [
                None,
                20.0,
                8.0,
                5.0,
                None,
            ],
            "agent_id": [
                "AGT-01",
                "AGT-02",
                "AGT-01",
                "AGT-02",
                "AGT-03",
            ],
            "customer_rating": [
                None,
                4.0,
                5.0,
                3.0,
                None,
            ],
            "issue_summary": [
                "Login problem",
                "Payment problem",
                "API problem",
                "Account question",
                "Invoice question",
            ],
        }
    )

    return engine


def test_reference_time():
    engine = make_engine()

    assert engine._reference_time() == pd.Timestamp(
        "2024-03-30 10:00:00"
    )


def test_basic_count_filter():
    engine = make_engine()

    plan = {
        "operation": "count",
        "target_column": None,
        "filters": [
            {
                "column": "status",
                "operator": "eq",
                "value": "Open",
            }
        ],
        "group_by": None,
        "sort": None,
        "limit": 10,
        "date_range": None,
        "special_condition": None,
        "special_condition_value": None,
    }

    result = engine.execute_plan(plan)

    assert result["count"] == 2


def test_average_customer_rating():
    engine = make_engine()

    plan = {
        "operation": "average",
        "target_column": "customer_rating",
        "filters": [
            {
                "column": "category",
                "operator": "eq",
                "value": "Technical",
            }
        ],
        "group_by": None,
        "sort": None,
        "limit": 10,
        "date_range": None,
        "special_condition": None,
        "special_condition_value": None,
    }

    result = engine.execute_plan(plan)

    assert result["average"] == 5.0


def test_group_by_count():
    engine = make_engine()

    plan = {
        "operation": "count",
        "target_column": None,
        "filters": [
            {
                "column": "status",
                "operator": "eq",
                "value": "Resolved",
            }
        ],
        "group_by": "agent_id",
        "sort": "desc",
        "limit": 1,
        "date_range": None,
        "special_condition": None,
        "special_condition_value": None,
    }

    result = engine.execute_plan(plan)

    assert result == {"AGT-02": 2}


def test_this_month_date_filter():
    engine = make_engine()

    plan = {
        "operation": "count",
        "target_column": None,
        "filters": [],
        "group_by": None,
        "sort": None,
        "limit": 10,
        "date_range": "this_month",
        "special_condition": None,
        "special_condition_value": None,
    }

    result = engine.execute_plan(plan)

    assert result["count"] == 4


def test_this_week_date_filter():
    engine = make_engine()

    plan = {
        "operation": "count",
        "target_column": None,
        "filters": [],
        "group_by": None,
        "sort": None,
        "limit": 10,
        "date_range": "this_week",
        "special_condition": None,
        "special_condition_value": None,
    }

    result = engine.execute_plan(plan)

    assert result["count"] == 3


def test_resolution_sla_breach_includes_unresolved():
    engine = make_engine()

    plan = {
        "operation": "list",
        "target_column": None,
        "filters": [
            {
                "column": "priority",
                "operator": "eq",
                "value": "Critical",
            }
        ],
        "group_by": None,
        "sort": None,
        "limit": 100,
        "date_range": None,
        "special_condition": (
            "resolution_sla_breach_hours"
        ),
        "special_condition_value": 12,
    }

    result = engine.execute_plan(plan)

    ticket_ids = {
        ticket["ticket_id"]
        for ticket in result["tickets"]
    }

    assert result["total_matches"] == 2
    assert "TKT-001" in ticket_ids
    assert "TKT-002" in ticket_ids
    assert "TKT-004" not in ticket_ids


def test_invalid_column_rejected():
    engine = make_engine()

    plan = {
        "operation": "count",
        "target_column": None,
        "filters": [
            {
                "column": "fake_column",
                "operator": "eq",
                "value": "test",
            }
        ],
        "group_by": None,
        "sort": None,
        "limit": 10,
        "date_range": None,
        "special_condition": None,
        "special_condition_value": None,
    }

    try:
        engine.execute_plan(plan)
        assert False, "Invalid column should be rejected."

    except ValueError:
        assert True