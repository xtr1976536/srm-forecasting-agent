from llm_agent import LLMForecastingAgent


def test_fallback_parser_and_validation():
    agent = LLMForecastingAgent()
    args = agent._validate(agent._fallback_parse("forecast AAPL MSFT horizon=5 K=20 cross asset"))
    assert args == {"tickers": ["AAPL", "MSFT"], "horizon": 5, "k": 20, "scope": "cross_asset", "criterion": "qlike"}


def test_invalid_request_is_rejected():
    agent = LLMForecastingAgent()
    try:
        agent._validate({"tickers": [], "horizon": 5, "k": 20, "scope": "cross_asset", "criterion": "qlike"})
    except ValueError as exc:
        assert "ticker" in str(exc).lower()
    else:
        raise AssertionError("empty ticker requests must be rejected")


def test_explanation_is_grounded_in_result():
    agent = LLMForecastingAgent()
    text = agent.explain({"task": {"tickers": ["AAPL"], "horizon": 5, "k": 20, "scope": "cross_asset"}, "predictions": {"AAPL": 0.123}})
    assert "AAPL" in text and "0.123" in text and "investment advice" in text
