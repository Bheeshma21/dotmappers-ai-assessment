import pandas as pd


class AnomalyDetector:
    """
    Detects unusual support tickets using statistical
    detection and business-risk rules.

    Anomalies include:
    1. Unusually high response time.
    2. Unusually high resolution time.
    3. Very low customer rating.
    4. Unresolved High/Critical tickets older than 24 hours.
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

    def _reference_time(self):
        """
        Use the latest timestamp in the supplied dataset
        as the reference time.

        This makes age-based anomaly detection meaningful
        for the historical assessment dataset.
        """

        latest_time = self.df["created_at"].max()

        if pd.isna(latest_time):
            raise ValueError(
                "Dataset does not contain valid created_at values."
            )

        return latest_time

    def detect_anomalies(self):
        df = self.df.copy()

        response_threshold = self._upper_iqr_threshold(
            df["response_time_hrs"]
        )

        resolution_threshold = self._upper_iqr_threshold(
            df["resolution_time_hrs"]
        )

        reference_time = self._reference_time()

        df["ticket_age_hrs"] = (
            reference_time - df["created_at"]
        ).dt.total_seconds() / 3600

        anomaly_reasons = []

        for _, row in df.iterrows():
            reasons = []

            response_time = row["response_time_hrs"]
            resolution_time = row["resolution_time_hrs"]
            rating = row["customer_rating"]
            ticket_age = row["ticket_age_hrs"]

            priority = str(row["priority"])
            status = str(row["status"])

            if (
                pd.notna(response_time)
                and response_time > response_threshold
            ):
                reasons.append(
                    f"High response time "
                    f"({response_time:.2f} hrs)"
                )

            if (
                pd.notna(resolution_time)
                and resolution_time > resolution_threshold
            ):
                reasons.append(
                    f"High resolution time "
                    f"({resolution_time:.2f} hrs)"
                )

            if (
                pd.notna(rating)
                and rating <= 2
            ):
                reasons.append(
                    f"Low customer rating "
                    f"({rating:.0f}/5)"
                )

            if (
                priority in ["High", "Critical"]
                and status != "Resolved"
                and pd.notna(ticket_age)
                and ticket_age > 24
            ):
                reasons.append(
                    "Unresolved high-priority ticket "
                    f"older than 24 hours "
                    f"({ticket_age:.2f} hrs old)"
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
                        self.df["response_time_hrs"]
                    )
                ),
                2,
            ),
            "resolution_time_threshold": round(
                float(
                    self._upper_iqr_threshold(
                        self.df["resolution_time_hrs"]
                    )
                ),
                2,
            ),
            "low_rating_threshold": 2,
            "unresolved_high_priority_age_threshold_hrs": 24,
        }

    def get_summary(self):
        anomalies = self.detect_anomalies()

        return {
            "total_tickets": int(len(self.df)),
            "anomaly_count": int(len(anomalies)),
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

    print("\nReference Time:")
    print(
        detector._reference_time()
    )

    print("\nThresholds:")
    print(
        detector.get_thresholds()
    )

    print("\nSummary:")
    print(
        detector.get_summary()
    )

    anomalies = detector.detect_anomalies()

    print("\nFirst 15 detected anomalies:")

    columns = [
        "ticket_id",
        "created_at",
        "category",
        "priority",
        "status",
        "response_time_hrs",
        "resolution_time_hrs",
        "ticket_age_hrs",
        "customer_rating",
        "anomaly_reasons",
    ]

    print(
        anomalies[columns]
        .head(15)
        .to_string(index=False)
    )