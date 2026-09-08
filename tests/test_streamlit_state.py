from pathlib import Path

def test_research_button_and_result_use_distinct_state_keys():
    text=Path("streamlit_app.py").read_text()
    assert 'key="submit_research"' in text
    assert 'st.session_state["research_result"]' in text
    assert 'st.session_state.research_run' not in text
