"""Auditable SRM forecasting agent."""
try:
    from .agent import SRMForecastingAgent
except ImportError:
    from agent import SRMForecastingAgent
__all__ = ["SRMForecastingAgent"]
