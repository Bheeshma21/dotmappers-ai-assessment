import pandas as pd


class AnomalyDetector:
    """
    Detects unusual support tickets.

    Detection methods:
    1. IQR-based unusually high response time.
    2. IQR-based unusually high resolution time.
    3. Very low customer rating (<= 2).
    4. High/Critical priority tickets that remain unresolved
       with unusually long response times.
    """

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    @staticmethod
    def _upper_iqr_threshold(series):
        clean_series = series.dropna()

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)

        iqr = q3 - q1

        return q3 + (1.5 * iqr)

    def detect_anomalies(self):
        df = self.df.copy()

        response_threshold = self._upper_iqr_threshold(
            df["response_time_hrs"]
        )

        resolution_threshold = self._upper_iqr_threshold(
            df["resolution_time_hrs"]
        )

        anomaly_reasons = []

        for _, row in df.iterrows():
            reasons = []

            response_time = row["response_time_hrs"]
            resolution_time = row["resolution_time_hrs"]
            rating = row["customer_rating"]

            priority = str(row["priority"])
            status = str(row["status"])

            # Statistical response-time anomaly
            if (
                pd.notna(response_time)
                and response_time > response_threshold
            ):
                reasons.append(
                    f"High response time "
                    f"({response_time:.2f} hrs)"
                )

            # Statistical resolution-time anomaly
            if (
                pd.notna(resolution_time)
                and resolution_time > resolution_threshold
            ):
                reasons.append(
                    f"High resolution time "
                    f"({resolution_time:.2f} hrs)"
                )

            # Customer-experience anomaly
            if (
                pd.notna(rating)
                and rating <= 2
            ):
                reasons.append(
                    f"Low customer rating "
                    f"({rating:.0f}/5)"
                )

            # Business-risk anomaly
            if (
                priority in ["High", "Critical"]
                and status != "Resolved"
                and pd.notna(response_time)
                and response_time > 4
            ):
                reasons.append(
                    "High-priority unresolved ticket "
                    f"with delayed response "
                    f"({response_time:.2f} hrs)"
                )

            anomaly_reasons.append(reasons)

        df["anomaly_reasons"] = anomaly_reasons

        df["is_anomaly"] = df[
            "anomaly_reasons"
        ].apply(
            lambda reasons: len(reasons) > 0
        )

        anomalies = df[
            df["is_anomaly"]
        ].copy()

        anomalies["anomaly_reasons"] = anomalies[
            "anomaly_reasons"
        ].apply(
            lambda reasons: "; ".join(reasons)
        )

        return anomalies

    def get_thresholds(self):
        return {
            "response_time_threshold": round(
                float(
                    self._upper_iqr_threshold(
                        self.df[
                            "response_time_hrs"
                        ]
                    )
                ),
                2,
            ),
            "resolution_time_threshold": round(
                float(
                    self._upper_iqr_threshold(
                        self.df[
                            "resolution_time_hrs"
                        ]
                    )
                ),
                2,
            ),
            "low_rating_threshold": 2,
            "high_priority_delayed_response_threshold": 4,
        }

    def get_summary(self):
        anomalies = self.detect_anomalies()

        return {
            "total_tickets": int(
                len(self.df)
            ),
            "anomaly_count": int(
                len(anomalies)
            ),
            "anomaly_percentage": round(
                (
                    len(anomalies)
                    / len(self.df)
                )
                * 100,
                2,
            ),
            "thresholds": self.get_thresholds(),
        }


if __name__ == "__main__":
    from data_engine import TicketDataEngine

    engine = TicketDataEngine()

    detector = AnomalyDetector(
        engine.get_dataframe()
    )

    print("=" * 60)
    print("DOTMAPPERS ANOMALY DETECTOR")
    print("=" * 60)

    print("\nThresholds:")
    print(
        detector.get_thresholds()
    )

    print("\nSummary:")
    print(
        detector.get_summary()
    )

    anomalies = (
        detector.detect_anomalies()
    )

    print(
        "\nFirst 15 detected anomalies:"
    )

    if anomalies.empty:
        print(
            "No anomalies detected."
        )

    else:
        columns = [
            "ticket_id",
            "category",
            "priority",
            "status",
            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating",
            "anomaly_reasons",
        ]

        print(
            anomalies[columns]
            .head(15)
            .to_string(index=False)
        )