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

    assert len(anomalies) == 79


def test_anomaly_summary():
    detector = get_detector()

    summary = detector.get_summary()

    assert summary["total_tickets"] == 500
    assert summary["anomaly_count"] == 79
    assert summary["anomaly_percentage"] == 15.8


def test_anomaly_thresholds():
    detector = get_detector()

    thresholds = detector.get_thresholds()

    assert thresholds["response_time_threshold"] == 7.65
    assert thresholds["resolution_time_threshold"] == 48.15
    assert thresholds["low_rating_threshold"] == 2

    assert (
        thresholds[
            "high_priority_delayed_response_threshold"
        ]
        == 4
    )


def test_anomaly_columns():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    assert "is_anomaly" in anomalies.columns
    assert "anomaly_reasons" in anomalies.columns


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


def test_high_priority_unresolved_detection():
    detector = get_detector()

    anomalies = detector.detect_anomalies()

    matching = anomalies[
        anomalies["anomaly_reasons"].str.contains(
            "High-priority unresolved ticket",
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