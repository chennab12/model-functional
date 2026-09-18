from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_fresh_app_session_has_no_exception():
    app=Path(__file__).parents[1]/"app.py"
    result=AppTest.from_file(str(app),default_timeout=20).run()
    assert not result.exception
