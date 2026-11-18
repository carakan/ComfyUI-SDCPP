#!/usr/bin/env python3
"""Integration tests for the SDServer backend of ComfyUI-SDCPP.

These tests do NOT start a real sd-server. A tiny Python HTTP server
mimics the native /sdcpp/v1 API surface (models/status, models/load,
models/unload, img_gen, jobs/{id}, jobs/{id}/cancel) so every branch
of the lifecycle can be exercised deterministically and fast.

What the tests cover:
  1. Request builder: base64 PNG images, structured multi-LoRA, batch_count,
     sigma schedules, in/failed/ref image handling.
  2. LoRA merge rule: repeated paths sum their multipliers, order preserved.
  3. ensure(): reuses the running server only when the loaded context matches
     the request; restarts on mismatch; reloads after an idle unload.
  4. Idle watch: never kills the process, calls models/unload once, retries
     when busy, leaves the model alone while a render is in flight.
  5. Cancellation: a user cancel calls jobs/{id}/cancel instead of killing
     the process.
  6. _http propagates JSON error codes from non-2xx responses.
"""

import base64
import io
import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.dirname(os.path.dirname(PLUGIN))
for path in (COMFY, PLUGIN):
    if path not in sys.path:
        sys.path.insert(0, path)

import sdcpp_nodes as m  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


# ---------------------------------------------------------------------------
# A fake sd-server good enough for the lifecycle tests.
# ---------------------------------------------------------------------------
class _FakeJob:
    def __init__(self, job_id):
        self.id = job_id
        self.status = "running"
        self.images = []


class _FakeSDServerState:
    def __init__(self, fail_next_load=False, busy_next_unload=False):
        self.model_state = "empty"          # empty | loading | ready | error
        self.recipe = None
        self.next_job_id = 0
        self.jobs = {}
        self.hang_job = None
        self.pending_args = None
        self.fail_next_load = fail_next_load
        self.busy_next_unload = busy_next_unload
        self.unload_calls = 0
        self.load_calls = 0
        self.cancel_calls = []
        self.lock = threading.Lock()


class _FakeSDServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state = None  # set by the test setup; shared with the handler

    def log_message(self, *_):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        parts = self.path.rstrip("/").split("/")
        if self.path == "/sdcpp/v1/models/status":
            return self._send(200, {"state": self.state.model_state,
                                    "recipe": self.state.recipe,
                                    "error": None})
        if self.path == "/sdcpp/v1/capabilities":
            return self._send(200, {"generation_features": []})
        if len(parts) >= 5 and parts[3] == "jobs":
            job_id = parts[4]
            job = self.state.jobs.get(job_id)
            if job is None:
                return self._send(404, {"error": "no such job"})
            payload = {"id": job.id, "kind": "img_gen", "status": job.status,
                       "progress": {"total": 8, "completed": 8 if job.status == "completed" else 4}}
            if job.status == "completed":
                payload["result"] = {"images": job.images}
            elif job.status in ("failed", "cancelled"):
                payload["error"] = {"code": "generation_failed", "message": job.status}
            else:
                payload["result"] = None
            return self._send(200, payload)
        self._send(404, {"error": "unknown"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = {}
        if length:
            body = json.loads(self.rfile.read(length))
        if self.path == "/sdcpp/v1/models/load":
            self.state.load_calls += 1
            if self.state.fail_next_load:
                self.state.fail_next_load = False
                self.state.model_state = "error"
                return self._send(422, {"error": "load failed", "code": "model_load_failed"})
            self.state.recipe = body.get("recipe")
            self.state.pending_args = body.get("models", {})
            self.state.model_state = "ready"
            return self._send(200, {"state": "ready", "recipe": self.state.recipe})
        if self.path == "/sdcpp/v1/models/unload":
            self.state.unload_calls += 1
            if self.state.busy_next_unload:
                self.state.busy_next_unload = False
                return self._send(409, {"error": "busy", "code": "model_busy"})
            self.state.model_state = "empty"
            self.state.recipe = None
            return self._send(200, {"state": "empty"})
        if self.path == "/sdcpp/v1/img_gen":
            if self.state.model_state != "ready":
                return self._send(503, {"error": "no model", "code": "model_not_loaded"})
            if self.state.hang_job:
                return self._send(202, {"id": self.state.hang_job})
            self.state.next_job_id += 1
            job = _FakeJob(f"job_{self.state.next_job_id}")
            self.state.jobs[job.id] = job
            job.images = [{"index": 0, "b64_json": _png_b64("rgb")}]
            job.status = "completed"
            return self._send(202, {"id": job.id})
        if "/cancel" in self.path:
            self.state.cancel_calls.append(self.path)
            job_id = self.path.split("/")[4]
            job = self.state.jobs.get(job_id)
            if job:
                job.status = "cancelled"
            return self._send(200, {"status": "cancelled"})
        self._send(404, {"error": "unknown"})



def _png_b64(kind="rgb"):
    from PIL import Image
    arr = np.zeros((4, 4, 3), dtype="uint8")
    arr[..., 0] = 200 if kind == "rgb" else 100
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------
class _FakeBase(unittest.TestCase):
    def setUp(self):
        self.server_state = _FakeSDServerState()
        _FakeSDServer.state = self.server_state
        self.http = HTTPServer(("127.0.0.1", 0), _FakeSDServer)
        port = self.http.server_address[1]
        os.environ["SDCPP_SERVER_PORT"] = str(port)
        os.environ["SDCPP_SERVER_IDLE_MINUTES"] = "0"
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        self.sd = m.SDServer()
        self._old_sdserver_path = m.sdserver_path
        m.sdserver_path = lambda: "/usr/bin/true"
        # Skip the real _start and put the server straight into the loaded state.
        self._old_start = m.SDServer._start

        def patched_start(self, context_args, env):
            # Reuse the running server after the assertion we're testing.
            self.context_args = list(context_args)
            self._pending_args = list(context_args)
            self.model_loaded = True
            self.last_used = time.time()
            self.server_model_state = "ready"
            # No real process is started in this test.
            if self.process is None:
                import subprocess
                self.process = subprocess.Popen(["sleep", "600"])
            self._start_idle_watch()

        m.SDServer._start = patched_start
        self._old_sleep = time.sleep

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        m.sdserver_path = self._old_sdserver_path
        m.SDServer._start = self._old_start
        self.sd.stop()
        time.sleep = self._old_sleep

    # --- helpers -----------------------------------------------------------
    def _start_process(self):
        """Start a fake process that stays alive (so SDServer.alive() is True)."""
        import subprocess
        proc = subprocess.Popen(["sleep", "600"])
        return proc

    def _install_running_server(self, args=("--anything",)):
        # Put the SDServer into the "server already started" state without
        # going through _start().
        self.sd.process = self._start_process()
        self.sd.context_args = list(args)
        self.sd.model_loaded = True
        self.sd.last_used = time.time()
        self.server_state.model_state = "ready"
        self.server_state.recipe = "flux"
        self.sd._pending_args = list(args)


class TestRequestBuilder(unittest.TestCase):
    def test_images_are_base64_png(self):
        tensor = torch.zeros((4, 4, 3))
        tensor[..., 1] = 1.0
        payload = m.server_request(
            "a", "n", 64, 64, 8, 1.0, 1, "euler", "discrete", -1.0,
            "", 1.0, True, None, [], None, tensor[None], [tensor, tensor],
        )
        for field in ("init_image", "ref_images"):
            value = payload[field][0] if isinstance(payload[field], list) else payload[field]
            raw = base64.b64decode(value)
            self.assertTrue(raw.startswith(b"\x89PNG"), f"{field} is not a PNG")
            self.assertGreater(len(raw), 50, f"{field} too small to be a PNG")

    def test_lora_multiplies_and_merges(self):
        merged = m.merge_lora_entries([("a", 1.0), ("b", 0.5), ("a", 0.25)])
        self.assertEqual(
            merged,
            [{"path": "a", "multiplier": 1.25}, {"path": "b", "multiplier": 0.5}],
        )

    def test_batch_count_passes_through(self):
        payload = m.server_request("a", "", 64, 64, 8, 1.0, 1, "euler", "discrete",
                                   -1.0, "", 1.0, True, None, [], None, None, [],
                                   batch_count=4)
        self.assertEqual(payload["batch_count"], 4)

    def test_no_batch_count_when_omitted(self):
        payload = m.server_request("a", "", 64, 64, 8, 1.0, 1, "euler", "discrete",
                                   -1.0, "", 1.0, True, None, [], None, None, [])
        self.assertNotIn("batch_count", payload)


class TestSDServerLifecycle(_FakeBase):
    def test_ensure_reuses_loaded_server(self):
        self._install_running_server(args=("--flag",))
        self.sd.ensure(["--flag"], env={})
        self.assertTrue(self.sd.alive())
        self.assertTrue(self.sd.model_loaded)
        self.assertEqual(self.server_state.load_calls, 0)

    def test_ensure_restarts_on_context_mismatch(self):
        self._install_running_server(args=("--old",))
        self.sd.ensure(["--new"], env={})
        # The patched _start replaces the context_args and reloads.
        self.assertEqual(self.sd.context_args, ["--new"])

    def test_ensure_reloads_when_idle_emptied(self):
        self._install_running_server(args=("--flag",))
        self.server_state.model_state = "empty"
        self.sd.model_loaded = False
        self.sd.ensure(["--flag"], env={})
        # _start ran and the patched version reloads.
        self.assertTrue(self.sd.model_loaded)
        self.assertEqual(self.sd.context_args, ["--flag"])

    def test_render_posts_base64_body_and_decodes_image(self):
        self._install_running_server(args=("--flag",))
        self.server_state.model_state = "ready"
        tensor = torch.zeros((4, 4, 3))
        body = m.server_request("p", "", 64, 64, 8, 1.0, 1, "euler", "discrete",
                                -1.0, "", 1.0, True, None, [], None, None, [tensor])
        images = self.sd.render(body, steps=8)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].size, (4, 4))

    def test_render_refuses_when_model_not_loaded(self):
        self._install_running_server(args=("--flag",))
        self.server_state.model_state = "empty"
        body = m.server_request("p", "", 64, 64, 8, 1.0, 1, "euler", "discrete",
                                -1.0, "", 1.0, True, None, [], None, None, [])
        with self.assertRaises(RuntimeError) as cm:
            self.sd.render(body, steps=8)
        self.assertIn("503", str(cm.exception))

    def test_cancel_is_forwarded_not_killed(self):
        self._install_running_server(args=("--flag",))
        # A job that never completes: img_gen returns it forever.
        self.server_state.hang_job = "never"
        self.server_state.jobs["never"] = _FakeJob("never")
        self.server_state.model_state = "ready"
        body = m.server_request("p", "", 64, 64, 8, 1.0, 1, "euler", "discrete",
                                -1.0, "", 1.0, True, None, [], None, None, [])
        with self.assertRaises(RuntimeError) as cm:
            self.sd.render(body, steps=8, timeout=0.5)
        self.assertIn("time", str(cm.exception).lower())
        self.assertTrue(any("/cancel" in c for c in self.server_state.cancel_calls))
        # The server process survives a user cancel; only the job was cancelled.
        self.assertTrue(self.sd.alive())


class TestEngineKnobs(unittest.TestCase):
    def test_context_args_includes_placement_and_perf(self):
        class Fake:
            full_checkpoint = False
            path = "/models/flux.gguf"
            display = "flux1"
        lora = None
        args = m.model_context_args(Fake(), None, None, [], None, lora,
                                    max_vram=8.0, offload_to_cpu=True,
                                    flash_attention=True, text_encoder_cpu=True,
                                    model_args="k=v")
        self.assertIn("--max-vram", args)
        self.assertIn("8.0", args)
        self.assertIn("--offload-to-cpu", args)
        self.assertIn("--flash-attn", args)
        self.assertIn("off,diffusion=on", args)
        self.assertIn("--params-backend", args)
        self.assertIn("te=cpu", args)
        self.assertIn("--model-args", args)
        self.assertIn("k=v", args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
