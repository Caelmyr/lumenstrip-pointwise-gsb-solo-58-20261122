"""Run the local workbench."""

from os import environ

from waitress import serve

from .http import create_app


if __name__ == "__main__":
    serve(create_app(), host=environ.get("HOST", "127.0.0.1"),
          port=int(environ.get("PORT", "19964")), threads=4, channel_timeout=180,
          max_request_body_size=30_000_000)
