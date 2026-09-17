import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="AI Support Ticket Analyst",
    page_icon="🤖",
    layout="wide",
)


st.title("🤖 AI Support Ticket Analyst")

st.write(
    "Ask natural-language questions about support tickets "
    "and explore ticket analytics and anomalies."
)


# --------------------------------------------------
# Check API connection
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    health_response.raise_for_status()

    health_data = health_response.json()

    st.success(
        f"API Connected • "
        f"{health_data['total_tickets']} tickets loaded"
    )

except requests.RequestException:
    st.error(
        "Cannot connect to the FastAPI server. "
        "Make sure Uvicorn is running."
    )

    st.stop()


# --------------------------------------------------
# Summary
# --------------------------------------------------

st.header("📊 Dataset Overview")

try:
    summary_response = requests.get(
        f"{API_URL}/summary",
        timeout=5,
    )

    summary_response.raise_for_status()

    summary = summary_response.json()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Tickets",
        summary["total_tickets"],
    )

    col2.metric(
        "Avg Response Time",
        f"{summary['average_response_time_hrs']} hrs",
    )

    col3.metric(
        "Avg Resolution Time",
        f"{summary['average_resolution_time_hrs']} hrs",
    )

    col4.metric(
        "Avg Customer Rating",
        f"{summary['average_customer_rating']}/5",
    )

except requests.RequestException as error:
    st.error(
        f"Could not load summary: {error}"
    )


# --------------------------------------------------
# AI Question Section
# --------------------------------------------------

st.header("💬 Ask the AI")

question = st.text_input(
    "Ask a question about the support tickets",
    placeholder=(
        "Example: Which agents have the highest "
        "average resolution time?"
    ),
)

if st.button(
    "Ask AI",
    type="primary",
):
    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    else:
        with st.spinner(
            "Analyzing support tickets..."
        ):
            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={
                        "question": question
                    },
                    timeout=60,
                )

                response.raise_for_status()

                result = response.json()

                st.subheader("AI Answer")

                st.markdown(
                    result["answer"]
                )

            except requests.RequestException as error:
                st.error(
                    f"AI request failed: {error}"
                )


# --------------------------------------------------
# Anomaly Section
# --------------------------------------------------

st.header("🚨 Anomaly Detection")

try:
    anomaly_response = requests.get(
        f"{API_URL}/anomalies",
        timeout=10,
    )

    anomaly_response.raise_for_status()

    anomaly_data = anomaly_response.json()

    anomaly_summary = anomaly_data["summary"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Detected Anomalies",
        anomaly_summary["anomaly_count"],
    )

    col2.metric(
        "Anomaly Percentage",
        f"{anomaly_summary['anomaly_percentage']}%",
    )

    with st.expander(
        "View detected anomalous tickets"
    ):
        st.dataframe(
            anomaly_data["anomalies"],
            width="stretch",
        )

except requests.RequestException as error:
    st.error(
        f"Could not load anomalies: {error}"
    )


st.divider()

st.caption(
    "Built with Python, FastAPI, Streamlit, "
    "Pandas and Groq LLM."
)