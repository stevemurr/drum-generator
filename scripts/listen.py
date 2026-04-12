"""
Simple audio preview server.

Usage:
    python listen.py [directory] [--port 8111]

Opens a browser page listing all audio files in the directory
with inline players.
"""

import argparse
import html
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote, unquote

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


def build_page(directory: str) -> str:
    files = []
    for f in sorted(os.listdir(directory)):
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
            files.append(f)

    rows = ""
    for f in files:
        safe_name = html.escape(f)
        safe_url = quote(f)
        rows += f"""
        <tr>
            <td class="name">{safe_name}</td>
            <td><audio controls preload="none" src="/audio/{safe_url}"></audio></td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Audio Preview — {html.escape(os.path.basename(os.path.abspath(directory)))}</title>
<style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #1a1a1a; color: #e0e0e0; }}
    h1 {{ font-size: 1.3rem; margin-bottom: 1rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    tr:hover {{ background: #2a2a2a; }}
    td {{ padding: 0.4rem 0.8rem; vertical-align: middle; }}
    .name {{ font-family: monospace; font-size: 0.85rem; white-space: nowrap; }}
    audio {{ height: 32px; }}
    .count {{ color: #888; font-size: 0.85rem; margin-bottom: 1rem; }}
</style>
</head>
<body>
    <h1>{html.escape(os.path.abspath(directory))}</h1>
    <div class="count">{len(files)} audio files</div>
    <table>
        {rows}
    </table>
</body>
</html>"""


class Handler(SimpleHTTPRequestHandler):
    directory = "."

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            page = build_page(self.server.audio_dir)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page.encode())
        elif self.path.startswith("/audio/"):
            filename = unquote(self.path[7:])
            filepath = os.path.join(self.server.audio_dir, filename)
            if not os.path.isfile(filepath):
                self.send_error(404)
                return
            ext = os.path.splitext(filename)[1].lower()
            mime = {
                ".wav": "audio/wav",
                ".mp3": "audio/mpeg",
                ".flac": "audio/flac",
                ".ogg": "audio/ogg",
                ".aif": "audio/aiff",
                ".aiff": "audio/aiff",
            }.get(ext, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", mime)
            size = os.path.getsize(filepath)
            self.send_header("Content-Length", str(size))
            self.end_headers()
            with open(filepath, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    parser = argparse.ArgumentParser(description="Audio preview server")
    parser.add_argument("directory", nargs="?", default=".", help="Directory with audio files")
    parser.add_argument("--port", type=int, default=8111)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.audio_dir = os.path.abspath(args.directory)
    print(f"Serving audio from {server.audio_dir}")
    print(f"http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
