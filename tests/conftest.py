"""Keep test runs out of LangSmith.

backend.config loads .env (which may switch tracing on) on import, so import it
first and then turn tracing off for the test process.
"""

import os

import backend.config  # noqa: F401

os.environ["LANGSMITH_TRACING"] = "false"
