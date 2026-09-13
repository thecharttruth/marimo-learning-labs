"""Run with: python3 -m unittest discover -s scripts -p 'test_*.py'."""
import importlib.util
from pathlib import Path
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

spec = importlib.util.spec_from_file_location("molab_sync", Path(__file__).with_name("molab-sync.py"))
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


class UrlValidationTests(unittest.TestCase):
    def test_accepts_molab_and_loopback(self):
        for url in ("https://sb-example.sb.molab.run/", "http://127.0.0.1:2718/", "http://[::1]:2718/", "http://localhost:2718/base"):
            with self.subTest(url=url):
                self.assertEqual(sync._validate_execution_url(url), url.rstrip("/"))

    def test_rejects_lookalikes_insecure_remote_urls_and_userinfo(self):
        for url in (
            "https://sb.molab.run.attacker.invalid/", "https://attacker.invalid/sb.molab.run",
            "http://sb-example.sb.molab.run", "https://sb-example.sb.molab.run:8443",
            "https://user:password@sb-example.sb.molab.run", "file:///etc/passwd",
            "https://sb-example.sb.molab.run/?next=bad", "https://sb-example.sb.molab.run/#fragment",
            "https://sb-example.sb.molab.run\\@attacker.invalid/", "https://sb-example.sb.molab.run\n",
        ):
            with self.subTest(url=url), self.assertRaises(SystemExit):
                sync._validate_execution_url(url)

    def test_cli_override_is_validated(self):
        with self.assertRaises(SystemExit):
            sync._execution_url({}, "https://attacker.invalid")


class RedirectTests(unittest.TestCase):
    def test_cross_origin_redirect_never_receives_token_but_same_origin_works(self):
        foreign_headers = []
        local_headers = []

        class Destination(BaseHTTPRequestHandler):
            def do_GET(self):
                foreign_headers.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)

        class Source(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/api/sessions":
                    self.send_response(302)
                    self.send_header("Location", f"http://localhost:{destination.server_port}/target")
                elif self.path == "/same-origin":
                    self.send_response(302)
                    self.send_header("Location", "/ok")
                else:
                    local_headers.append(self.headers.get("Authorization"))
                    self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        source = ThreadingHTTPServer(("127.0.0.1", 0), Source)
        servers = [destination, source]
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{source.server_port}"
            status, _ = sync._http_get(url, "dummy-test-value", "/api/sessions")
            self.assertEqual(status, 302)
            self.assertEqual(foreign_headers, [])
            status, _ = sync._http_get(url, "dummy-test-value", "/same-origin")
            self.assertEqual(status, 200)
            self.assertEqual(local_headers, ["Bearer dummy-test-value"])
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
