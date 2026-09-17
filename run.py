import subprocess
import sys
import time


def main():
    print("=" * 60)
    print("DOTMAPPERS AI SUPPORT TICKET ANALYST")
    print("=" * 60)

    print("\nStarting FastAPI backend...")

    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ]
    )

    time.sleep(3)

    if api_process.poll() is not None:
        print(
            "\nFastAPI failed to start."
        )
        return

    print("FastAPI started successfully.")
    print(
        "API: http://127.0.0.1:8000"
    )

    print("\nStarting Streamlit...")

    try:
        streamlit_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "app.py",
            ]
        )

        streamlit_process.wait()

    except KeyboardInterrupt:
        print(
            "\nStopping application..."
        )

    finally:
        if api_process.poll() is None:
            api_process.terminate()

        print(
            "Application stopped."
        )


if __name__ == "__main__":
    main()