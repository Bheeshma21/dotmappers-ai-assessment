from data_engine import TicketDataEngine


def test_dataset_loads():
    engine = TicketDataEngine()

    assert engine.get_total_tickets() == 500


def test_required_columns_exist():
    engine = TicketDataEngine()

    expected_columns = [
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

    for column in expected_columns:
        assert column in engine.get_columns()


def test_categories():
    engine = TicketDataEngine()

    assert engine.get_categories() == [
        "Billing",
        "General",
        "Technical",
    ]


def test_priorities():
    engine = TicketDataEngine()

    assert engine.get_priorities() == [
        "Critical",
        "High",
        "Low",
        "Medium",
    ]


def test_statuses():
    engine = TicketDataEngine()

    assert engine.get_statuses() == [
        "Escalated",
        "Open",
        "Resolved",
    ]


def test_summary():
    engine = TicketDataEngine()

    summary = engine.get_summary()

    assert summary["total_tickets"] == 500
    assert summary["average_response_time_hrs"] == 2.62
    assert summary["average_resolution_time_hrs"] == 19.16
    assert summary["average_customer_rating"] == 3.75


def test_analysis_context():
    engine = TicketDataEngine()

    context = engine.build_analysis_context()

    assert "Total number of tickets: 500" in context
    assert "Critical / Escalated: 22 tickets" in context
    assert "Technical: 20.59 hours" in context
    assert "AGT-08: 26.50 hours" in context