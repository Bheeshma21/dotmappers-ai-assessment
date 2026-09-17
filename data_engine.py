from pathlib import Path

import pandas as pd


class TicketDataEngine:
    """Loads, validates, and analyzes the support ticket dataset."""

    REQUIRED_COLUMNS = [
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

    def __init__(self, file_path="data/support_tickets.csv"):
        self.file_path = Path(file_path)
        self.df = self._load_data()

    def _load_data(self):
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Dataset is missing columns: {missing_columns}"
            )

        # Convert date column
        df["created_at"] = pd.to_datetime(
            df["created_at"],
            errors="coerce",
        )

        # Convert numerical columns
        numeric_columns = [
            "response_time_hrs",
            "resolution_time_hrs",
            "customer_rating",
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        return df

    def get_dataframe(self):
        return self.df.copy()

    def get_total_tickets(self):
        return int(len(self.df))

    def get_columns(self):
        return self.df.columns.tolist()

    def get_categories(self):
        return sorted(
            self.df["category"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    def get_priorities(self):
        return sorted(
            self.df["priority"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    def get_statuses(self):
        return sorted(
            self.df["status"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    def get_agents(self):
        return sorted(
            self.df["agent_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    def get_summary(self):
        return {
            "total_tickets": self.get_total_tickets(),
            "categories": self.get_categories(),
            "priorities": self.get_priorities(),
            "statuses": self.get_statuses(),
            "agents": self.get_agents(),
            "average_response_time_hrs": round(
                float(
                    self.df["response_time_hrs"].mean()
                ),
                2,
            ),
            "average_resolution_time_hrs": round(
                float(
                    self.df["resolution_time_hrs"].mean()
                ),
                2,
            ),
            "average_customer_rating": round(
                float(
                    self.df["customer_rating"].mean()
                ),
                2,
            ),
        }

    def build_analysis_context(self):
        """
        Build calculated context from the real CSV.

        Python/Pandas calculates the statistics first.
        The LLM receives these calculated values so it
        does not need to invent or calculate numbers.
        """

        df = self.df

        lines = []

        # --------------------------------------------------
        # Overall statistics
        # --------------------------------------------------

        lines.append(
            f"Total number of tickets: {len(df)}"
        )

        lines.append(
            "Overall average response time: "
            f"{df['response_time_hrs'].mean():.2f} hours"
        )

        lines.append(
            "Overall average resolution time: "
            f"{df['resolution_time_hrs'].mean():.2f} hours"
        )

        lines.append(
            "Overall average customer rating: "
            f"{df['customer_rating'].mean():.2f}/5"
        )

        # --------------------------------------------------
        # Category counts
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by category:"
        )

        category_counts = (
            df["category"]
            .value_counts()
            .sort_values(ascending=False)
        )

        for category, count in category_counts.items():
            lines.append(
                f"- {category}: {int(count)} tickets"
            )

        # --------------------------------------------------
        # Priority counts
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by priority:"
        )

        priority_counts = (
            df["priority"]
            .value_counts()
            .sort_values(ascending=False)
        )

        for priority, count in priority_counts.items():
            lines.append(
                f"- {priority}: {int(count)} tickets"
            )

        # --------------------------------------------------
        # Status counts
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by status:"
        )

        status_counts = (
            df["status"]
            .value_counts()
            .sort_values(ascending=False)
        )

        for status, count in status_counts.items():
            lines.append(
                f"- {status}: {int(count)} tickets"
            )

        # --------------------------------------------------
        # Agent ticket counts
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by agent:"
        )

        agent_counts = (
            df["agent_id"]
            .value_counts()
            .sort_values(ascending=False)
        )

        for agent, count in agent_counts.items():
            lines.append(
                f"- {agent}: {int(count)} tickets"
            )

        # --------------------------------------------------
        # Agent resolution times
        # --------------------------------------------------

        lines.append(
            "\nAverage resolution time by agent:"
        )

        agent_resolution = (
            df.groupby("agent_id")[
                "resolution_time_hrs"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for agent, value in agent_resolution.items():
            lines.append(
                f"- {agent}: {value:.2f} hours"
            )

        # --------------------------------------------------
        # Agent response times
        # --------------------------------------------------

        lines.append(
            "\nAverage response time by agent:"
        )

        agent_response = (
            df.groupby("agent_id")[
                "response_time_hrs"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for agent, value in agent_response.items():
            lines.append(
                f"- {agent}: {value:.2f} hours"
            )

        # --------------------------------------------------
        # Agent ratings
        # --------------------------------------------------

        lines.append(
            "\nAverage customer rating by agent:"
        )

        agent_rating = (
            df.groupby("agent_id")[
                "customer_rating"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for agent, value in agent_rating.items():
            lines.append(
                f"- {agent}: {value:.2f}/5"
            )

        # --------------------------------------------------
        # Category resolution time
        # --------------------------------------------------

        lines.append(
            "\nAverage resolution time by category:"
        )

        category_resolution = (
            df.groupby("category")[
                "resolution_time_hrs"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for category, value in category_resolution.items():
            lines.append(
                f"- {category}: {value:.2f} hours"
            )

        # --------------------------------------------------
        # Category response time
        # --------------------------------------------------

        lines.append(
            "\nAverage response time by category:"
        )

        category_response = (
            df.groupby("category")[
                "response_time_hrs"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for category, value in category_response.items():
            lines.append(
                f"- {category}: {value:.2f} hours"
            )

        # --------------------------------------------------
        # Category ratings
        # --------------------------------------------------

        lines.append(
            "\nAverage customer rating by category:"
        )

        category_rating = (
            df.groupby("category")[
                "customer_rating"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for category, value in category_rating.items():
            lines.append(
                f"- {category}: {value:.2f}/5"
            )

        # --------------------------------------------------
        # Priority resolution time
        # --------------------------------------------------

        lines.append(
            "\nAverage resolution time by priority:"
        )

        priority_resolution = (
            df.groupby("priority")[
                "resolution_time_hrs"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for priority, value in priority_resolution.items():
            lines.append(
                f"- {priority}: {value:.2f} hours"
            )

        # --------------------------------------------------
        # Priority ratings
        # --------------------------------------------------

        lines.append(
            "\nAverage customer rating by priority:"
        )

        priority_rating = (
            df.groupby("priority")[
                "customer_rating"
            ]
            .mean()
            .sort_values(ascending=False)
        )

        for priority, value in priority_rating.items():
            lines.append(
                f"- {priority}: {value:.2f}/5"
            )

        # --------------------------------------------------
        # Category + Status combinations
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by category and status:"
        )

        category_status = (
            df.groupby(
                ["category", "status"]
            )
            .size()
            .reset_index(name="count")
        )

        for _, row in category_status.iterrows():
            lines.append(
                f"- {row['category']} / "
                f"{row['status']}: "
                f"{int(row['count'])} tickets"
            )

        # --------------------------------------------------
        # Priority + Status combinations
        # --------------------------------------------------

        lines.append(
            "\nTicket counts by priority and status:"
        )

        priority_status = (
            df.groupby(
                ["priority", "status"]
            )
            .size()
            .reset_index(name="count")
        )

        for _, row in priority_status.iterrows():
            lines.append(
                f"- {row['priority']} / "
                f"{row['status']}: "
                f"{int(row['count'])} tickets"
            )

        return "\n".join(lines)


if __name__ == "__main__":
    engine = TicketDataEngine()

    print("=" * 60)
    print("DOTMAPPERS REAL DATA ANALYSIS")
    print("=" * 60)

    print(
        engine.build_analysis_context()
    )