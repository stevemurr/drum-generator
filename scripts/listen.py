"""
Audio preview server with folder upload.

Usage:
    python listen.py [directory] [--port 8111]

Browse and listen to audio files. Upload folders (zipped or drag-and-drop)
from your local machine to the server's directory.
"""

import argparse
import html
import io
import json
import os
import zipfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote, unquote

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff"}


def build_page(directory: str) -> str:
    files = []
    for root, _, filenames in os.walk(directory):
        for f in sorted(filenames):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS:
                rel = os.path.relpath(os.path.join(root, f), directory)
                files.append(rel)

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
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #1a1a1a; color: #e0e0e0; }}
    h1 {{ font-size: 1.3rem; margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    tr:hover {{ background: #2a2a2a; }}
    td {{ padding: 0.4rem 0.8rem; vertical-align: middle; }}
    .name {{ font-family: monospace; font-size: 0.85rem; white-space: nowrap; }}
    audio {{ height: 32px; }}
    .count {{ color: #888; font-size: 0.85rem; margin-bottom: 1rem; }}

    .upload-zone {{
        border: 2px dashed #444; border-radius: 8px; padding: 1.5rem;
        text-align: center; margin-bottom: 1.5rem; cursor: pointer;
        transition: border-color 0.2s, background 0.2s;
    }}
    .upload-zone:hover, .upload-zone.dragover {{
        border-color: #888; background: #222;
    }}
    .upload-zone input {{ display: none; }}
    .upload-zone p {{ margin: 0; color: #888; font-size: 0.9rem; }}
    .upload-zone .hint {{ font-size: 0.8rem; color: #555; margin-top: 0.3rem; }}
    .upload-status {{
        font-size: 0.85rem; color: #aaa; margin-bottom: 1rem; min-height: 1.2em;
    }}
    .progress-bar {{
        width: 100%; height: 4px; background: #333; border-radius: 2px;
        margin-bottom: 1rem; overflow: hidden; display: none;
    }}
    .progress-bar .fill {{
        height: 100%; background: #5b8; width: 0%; transition: width 0.2s;
    }}
</style>
</head>
<body>
    <h1>{html.escape(os.path.abspath(directory))}</h1>

    <div class="upload-zone" id="dropzone">
        <p>Drop audio files or a .zip here, or click to browse</p>
        <p class="hint">Accepts {', '.join(sorted(AUDIO_EXTS))} — folders via .zip</p>
        <input type="file" id="fileinput" multiple>
    </div>
    <div class="progress-bar" id="progressbar"><div class="fill" id="progressfill"></div></div>
    <div class="upload-status" id="status"></div>

    <div class="count">{len(files)} audio files</div>
    <table>
        {rows}
    </table>

<script>
const dropzone = document.getElementById('dropzone');
const fileinput = document.getElementById('fileinput');
const status = document.getElementById('status');
const progressbar = document.getElementById('progressbar');
const progressfill = document.getElementById('progressfill');

dropzone.addEventListener('click', () => fileinput.click());
dropzone.addEventListener('dragover', e => {{ e.preventDefault(); dropzone.classList.add('dragover'); }});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', e => {{
    e.preventDefault();
    dropzone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
}});
fileinput.addEventListener('change', () => handleFiles(fileinput.files));

async function handleFiles(fileList) {{
    const files = Array.from(fileList);
    if (!files.length) return;

    let total = files.length;
    let done = 0;
    progressbar.style.display = 'block';
    progressfill.style.width = '0%';

    for (const file of files) {{
        status.textContent = `Uploading ${{file.name}}...`;
        const formData = new FormData();
        formData.append('file', file);

        try {{
            const resp = await fetch('/upload', {{ method: 'POST', body: formData }});
            const result = await resp.json();
            if (!resp.ok) {{
                status.textContent = `Error: ${{result.error}}`;
                return;
            }}
            done++;
            progressfill.style.width = (done / total * 100) + '%';
        }} catch (err) {{
            status.textContent = `Upload failed: ${{err}}`;
            return;
        }}
    }}

    status.textContent = `Uploaded ${{done}} file(s). Refreshing...`;
    setTimeout(() => location.reload(), 500);
}}
</script>
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
            # Prevent path traversal
            filepath = os.path.realpath(filepath)
            if not filepath.startswith(os.path.realpath(self.server.audio_dir)):
                self.send_error(403)
                return
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

    def do_POST(self):
        if self.path != "/upload":
            self.send_error(404)
            return

        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._json_response(400, {"error": "Expected multipart/form-data"})
            return

        # Save the raw upload to a temp file, then figure out what it is
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Parse boundary (may have quotes)
        boundary = content_type.split("boundary=")[-1].strip().strip('"').encode()

        # Extract file from multipart
        filename, filedata = self._parse_multipart(body, boundary)
        if not filename or not filedata:
            self._json_response(400, {"error": "No file in request"})
            return

        ext = os.path.splitext(filename)[1].lower()
        print(f"[upload] received: {filename} ({len(filedata)} bytes, ext={ext})")

        if ext == ".zip":
            count = self._handle_zip(filedata)
            self._json_response(200, {"status": "ok", "extracted": count})
        elif ext in AUDIO_EXTS:
            safe_name = os.path.basename(filename)
            dest = os.path.join(self.server.audio_dir, safe_name)
            with open(dest, "wb") as f:
                f.write(filedata)
            print(f"[upload] saved {safe_name}")
            self._json_response(200, {"status": "ok", "file": safe_name})
        else:
            self._json_response(400, {"error": f"Unsupported file type: {ext}"})

    def _handle_zip(self, data: bytes) -> int:
        """Extract audio files from a zip archive."""
        count = 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                ext = os.path.splitext(info.filename)[1].lower()
                if ext not in AUDIO_EXTS:
                    continue
                # Flatten into audio_dir (use basename to avoid nested paths)
                safe_name = os.path.basename(info.filename)
                if not safe_name:
                    continue
                dest = os.path.join(self.server.audio_dir, safe_name)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    dst.write(src.read())
                count += 1
                print(f"[upload/zip] {safe_name}")
        return count

    def _parse_multipart(self, body: bytes, boundary: bytes) -> tuple[str | None, bytes | None]:
        """Minimal multipart parser — extracts first file.

        Multipart format: each part is between --boundary markers,
        with headers separated from body by \\r\\n\\r\\n. The body
        ends with \\r\\n before the next --boundary.
        """
        delim = b"--" + boundary
        # Find the part containing a file
        start = body.find(delim)
        while start >= 0:
            # Find end of this part (next boundary)
            part_start = start + len(delim)
            part_end = body.find(delim, part_start)
            if part_end < 0:
                part_end = len(body)

            part = body[part_start:part_end]

            if b"filename=" in part:
                header_end = part.find(b"\r\n\r\n")
                if header_end < 0:
                    start = part_end
                    continue
                header = part[:header_end].decode(errors="replace")
                filedata = part[header_end + 4:]
                # Remove the trailing \r\n that precedes the next boundary
                if filedata.endswith(b"\r\n"):
                    filedata = filedata[:-2]

                for segment in header.split(";"):
                    segment = segment.strip()
                    if segment.startswith("filename="):
                        filename = segment.split("=", 1)[1].strip('" ')
                        return filename, filedata

            start = part_end

        return None, None

    def _json_response(self, code: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Audio preview server with upload")
    parser.add_argument("directory", nargs="?", default=".", help="Directory for audio files")
    parser.add_argument("--port", type=int, default=8111)
    args = parser.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    server.audio_dir = os.path.abspath(args.directory)
    print(f"Serving audio from {server.audio_dir}")
    print(f"http://localhost:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
