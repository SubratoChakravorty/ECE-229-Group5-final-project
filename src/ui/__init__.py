import dash
import dash_bootstrap_components as dbc
from flask_caching import Cache

from src.config import cache_dir


# Initialize app and cache
app = dash.Dash(__name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}],
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR' : cache_dir,
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
