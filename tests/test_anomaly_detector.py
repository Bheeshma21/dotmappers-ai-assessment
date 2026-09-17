from anomaly_detector import AnomalyDetector
from data_engine import TicketDataEngine


def get_detector():
    engine = TicketDataEngine()

    return AnomalyDetector(
        engine.get_dataframe()
    )


def test_anomaly_count():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    assert len(anomalies) == 145


def test_anomaly_summary():
    detector = get_detector()

    summary = detector.get_summary()

    assert summary["total_tickets"] == 500
    assert summary["anomaly_count"] == 145
    assert summary["anomaly_percentage"] == 29.0


def test_anomaly_thresholds():
    detector = get_detector()

    thresholds = detector.get_thresholds()

    assert thresholds["response_time_threshold"] == 7.65
    assert thresholds["resolution_time_threshold"] == 48.15
    assert thresholds["low_rating_threshold"] == 2

    assert (
        thresholds[
            "unresolved_high_priority_age_threshold_hrs"
        ]
        == 24
    )


def test_anomaly_columns():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    assert "is_anomaly" in anomalies.columns
    assert "anomaly_reasons" in anomalies.columns
    assert "ticket_age_hrs" in anomalies.columns


def test_anomaly_reasons_not_empty():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    assert anomalies["anomaly_reasons"].notna().all()

    assert (
        anomalies["anomaly_reasons"]
        .str.strip()
        .ne("")
        .all()
    )


def test_unresolved_high_priority_detection():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    matching = anomalies[
        anomalies["anomaly_reasons"].str.contains(
            "Unresolved high-priority ticket "
            "older than 24 hours",
            regex=False,
        )
    ]

    assert len(matching) > 0

    assert matching["priority"].isin(
        ["High", "Critical"]
    ).all()

    assert (
        matching["status"] != "Resolved"
    ).all()

    assert (
        matching["ticket_age_hrs"] > 24
    ).all()


def test_reference_time():
    detector = get_detector()

    reference_time = detector._reference_time()

    assert str(reference_time) == (
        "2024-03-30 18:06:00"
    )