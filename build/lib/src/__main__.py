from .ui import app
# noinspection PyUnresolvedReferences
import src.ui.dashboard
import os

app.run_server(debug=False,
               dev_tools_hot_reload=False,
               host=os.getenv("HOST", "192.168.1.10"),
               port=os.getenv("PORT", "8050"),
               )
