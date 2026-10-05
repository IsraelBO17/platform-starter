# tests/test_scripts.py
import os

os.environ.setdefault("GROQ_API_KEY", "test")


def test_scripts_import():
    import scripts.compare_models  # noqa: F401
    import scripts.compare_prompts  # noqa: F401
    import scripts.first_call  # noqa: F401
    import scripts.try_triage  # noqa: F401
