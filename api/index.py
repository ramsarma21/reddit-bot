# api/index.py
# Vercel looks for a module in /api and uses the exported "app" object.

from app import app as app  # re-export your Flask instance from app.py
# If your file is named differently, adjust the import accordingly.
