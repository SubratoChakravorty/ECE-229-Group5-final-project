from .ui import app
# noinspection PyUnresolvedReferences
import src.ui.dashboard

app.run_server(debug=True, dev_tools_hot_reload=False)
