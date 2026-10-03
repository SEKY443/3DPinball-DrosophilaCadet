"""Serve web/ for local / LAN testing.

`python -m http.server` keeps the default listen backlog of 5, so a browser that opens many parallel connections
(the page loads ~12 ES modules plus the WASM build at once) gets ERR_CONNECTION_RESET on some of them and the page
fails to start. This server is the same SimpleHTTPRequestHandler with a larger backlog and threads.

Usage: python3 scripts/serve_web.py [--bind 127.0.0.1] [--port 4242]
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os


class Server(http.server.ThreadingHTTPServer):
	request_queue_size = 128
	daemon_threads = True


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	ap.add_argument("--bind", default="127.0.0.1")
	ap.add_argument("--port", type=int, default=4242)
	args = ap.parse_args()
	root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")
	handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=os.path.abspath(root))
	with Server((args.bind, args.port), handler) as httpd:
		print(f"serving {os.path.abspath(root)} on http://{args.bind}:{args.port}/", flush=True)
		httpd.serve_forever()


if __name__ == "__main__":
	main()
