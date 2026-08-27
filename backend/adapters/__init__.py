from backend.adapters.base import Adapter, AdapterResult
from backend.adapters.lodging import LodgingAdapter
from backend.adapters.transport import TransportAdapter
from backend.adapters.weather import WeatherAdapter

__all__ = [
    "Adapter",
    "AdapterResult",
    "LodgingAdapter",
    "TransportAdapter",
    "WeatherAdapter",
]
