#!/usr/bin/env python3
"""Regression tests for the bridge's transport and Max dispatch boundaries."""

from __future__ import annotations

import base64
import json
import pathlib
import subprocess
import sys
import unittest

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import ableton_udp_bridge as bridge


M4L_DIR = pathlib.Path(__file__).with_name("m4l")
PATCH_PATH = M4L_DIR / "LiveUdpBridge.maxpat"
BRIDGE_JS_PATH = M4L_DIR / "live_udp_bridge.js"
RECEIVER_JS_PATH = M4L_DIR / "osc_loopback_receiver.js"
TEST_AUTH_TOKEN = "test-auth-token-0123456789"


def _run_bridge_js(body: str) -> object:
    harness = f"""
const fs = require("node:fs");
const vm = require("node:vm");
const source = fs.readFileSync({json.dumps(str(BRIDGE_JS_PATH))}, "utf8");
const context = {{
  post: () => {{}},
  outlet: () => {{}},
  ack: () => {{}},
  arrayfromargs: (args) => Array.from(args),
  Dict: function Dict() {{}},
  LiveAPI: function LiveAPI() {{}},
  inlet: 1,
}};
vm.createContext(context);
vm.runInContext(source, context);
const result = (() => {{
{body}
}})();
process.stdout.write(JSON.stringify(result));
"""
    completed = subprocess.run(
        ["node", "-e", harness],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return json.loads(completed.stdout)


class SecurityBoundaryTests(unittest.TestCase):
    def test_max_patch_isolates_network_dispatch_and_token_setup(self) -> None:
        patch = json.loads(PATCH_PATH.read_text())
        boxes = [item["box"] for item in patch["patcher"]["boxes"]]
        lines = [item["patchline"] for item in patch["patcher"]["lines"]]

        receiver = next(
            box
            for box in boxes
            if str(box.get("text", "")).startswith(
                "node.script osc_loopback_receiver.js"
            )
        )
        self.assertIn("@restart 1", receiver["text"])
        limiter = next(box for box in boxes if box.get("text") == "qlim 10 @defer 1")
        route = next(
            box for box in boxes if str(box.get("text", "")).startswith("route ")
        )
        dispatcher = next(
            box for box in boxes if box.get("text") == "prepend osc_dispatch"
        )
        js_box = next(box for box in boxes if box.get("text") == "js live_udp_bridge.js")
        auth_box = next(
            box
            for box in boxes
            if str(box.get("text", "")).startswith("set_auth_token ")
        )
        fallback_outlet = len(str(route["text"]).split()[1:])

        self.assertFalse(
            any(str(box.get("text", "")).startswith("udpreceive") for box in boxes)
        )
        receiver_rect = receiver["patching_rect"]
        limiter_rect = limiter["patching_rect"]
        self.assertLessEqual(
            receiver_rect[0] + receiver_rect[2],
            limiter_rect[0],
            "the loopback receiver and qlim objects must remain visually distinct",
        )
        self.assertEqual(js_box["numinlets"], 2)
        self.assertIn(
            "osc_loopback_receiver.js",
            {item["name"] for item in patch["patcher"]["dependency_cache"]},
        )
        self.assertIn(
            {"source": [receiver["id"], 0], "destination": [limiter["id"], 0]},
            [
                {"source": line.get("source"), "destination": line.get("destination")}
                for line in lines
            ],
        )
        self.assertTrue(
            any(
                line.get("source") == [route["id"], fallback_outlet]
                and line.get("destination") == [dispatcher["id"], 0]
                for line in lines
            )
        )
        self.assertFalse(
            any(
                line.get("source") == [route["id"], fallback_outlet]
                and line.get("destination", [None])[0] == js_box["id"]
                for line in lines
            )
        )
        self.assertTrue(
            any(
                line.get("source") == [dispatcher["id"], 0]
                and line.get("destination") == [js_box["id"], 0]
                for line in lines
            )
        )
        token_inputs = [
            line
            for line in lines
            if line.get("destination") == [js_box["id"], 1]
        ]
        self.assertEqual(len(token_inputs), 1)
        self.assertEqual(token_inputs[0].get("source"), [auth_box["id"], 0])

    def test_js_rejects_network_token_setup_and_named_helpers(self) -> None:
        result = _run_bridge_js(
            f"""
const wrapperCalls = [];
context.inlet = 1;
context.set_auth_token({json.dumps(TEST_AUTH_TOKEN)});
context.API_FALLBACK_HANDLERS.api_session_context = (...args) => wrapperCalls.push(args);
context.inlet = 0;
context.osc_dispatch("set_auth_token", "attacker-token-0123456789");
context.osc_dispatch("renameTrack", 0, "Owned");
context.osc_dispatch("/unknown_selector", "ignored");
context.osc_dispatch("/api/session_context", "req-context");
return {{ token: context.bridgeAuthToken, wrapperCalls }};
"""
        )

        self.assertEqual(result["token"], TEST_AUTH_TOKEN)
        self.assertEqual(result["wrapperCalls"], [["req-context"]])

    def test_js_fallback_rejects_inherited_object_properties(self) -> None:
        inherited_selectors = [
            "/constructor",
            "/toString",
            "/valueOf",
            "/hasOwnProperty",
            "/__defineGetter__",
            "/__lookupGetter__",
            "/__proto__",
        ]
        result = _run_bridge_js(
            f"""
const outputs = [];
const wrapperCalls = [];
context.outlet = (...args) => outputs.push(args);
context.API_FALLBACK_HANDLERS.api_session_context = (...args) => wrapperCalls.push(args);
{json.dumps(inherited_selectors)}.forEach((selector) => {{
  context.osc_dispatch(selector, "req-inherited");
}});
context.osc_dispatch("/api/session_context", "req-context");
return {{
  acks: outputs.filter((args) => args[1] === "/ack"),
  wrapperCalls,
}};
"""
        )

        self.assertEqual(
            result["acks"],
            [
                [
                    0,
                    "/ack",
                    "error",
                    "unknown_selector",
                    selector,
                    "request_correlation",
                    "req:",
                ]
                for selector in inherited_selectors
            ],
        )
        self.assertEqual(result["wrapperCalls"], [["req-context"]])

    def test_js_token_setup_requires_local_inlet(self) -> None:
        result = _run_bridge_js(
            f"""
context.inlet = 0;
context.set_auth_token("attacker-token-0123456789");
const afterNetworkAttempt = context.bridgeAuthToken;
context.inlet = 1;
context.set_auth_token({json.dumps(TEST_AUTH_TOKEN)});
const afterLocalSetup = context.bridgeAuthToken;
context.inlet = 0;
context.set_auth_token("replacement-token-0123456789");
return {{ afterNetworkAttempt, afterLocalSetup, finalToken: context.bridgeAuthToken }};
"""
        )

        self.assertEqual(result["afterNetworkAttempt"], "")
        self.assertEqual(result["afterLocalSetup"], TEST_AUTH_TOKEN)
        self.assertEqual(result["finalToken"], TEST_AUTH_TOKEN)

    def test_node_receiver_decodes_python_client_packets(self) -> None:
        packet = bridge.encode_osc_message("/probe", (-2, 1.25, "hello"))
        script = f"""
const receiver = require({json.dumps(str(RECEIVER_JS_PATH))});
const decoded = receiver.decodeOscMessage(
  Buffer.from({json.dumps(base64.b64encode(packet).decode())}, "base64")
);
process.stdout.write(JSON.stringify(decoded));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {"address": "/probe", "args": [-2, 1.25, "hello"]},
        )

    def test_node_receiver_rejects_malformed_packets(self) -> None:
        valid = bridge.encode_osc_message("/probe", (1,))
        packets = [b"#bundle\x00", valid + b"\x00\x00\x00\x00", b"bad"]
        encoded = [base64.b64encode(packet).decode() for packet in packets]
        script = f"""
const receiver = require({json.dumps(str(RECEIVER_JS_PATH))});
const results = {json.dumps(encoded)}.map((value) => {{
  try {{
    receiver.decodeOscMessage(Buffer.from(value, "base64"));
    return "accepted";
  }} catch (error) {{
    return error.message;
  }}
}});
process.stdout.write(JSON.stringify(results));
"""
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        results = json.loads(completed.stdout)

        self.assertTrue(all(result != "accepted" for result in results))
        self.assertIn("bundles", results[0])
        self.assertIn("trailing bytes", results[1])

    def test_node_receiver_starts_when_imported_by_node_for_max(self) -> None:
        script = f"""
const {{ EventEmitter }} = require("node:events");
const Module = require("node:module");
const {{ pathToFileURL }} = require("node:url");
const receiverPath = {json.dumps(str(RECEIVER_JS_PATH))};
const bindings = [];
const originalLoad = Module._load;

Module._load = function(request, parent, isMain) {{
  if (request === "node:dgram") {{
    return {{
      createSocket: () => {{
        const server = new EventEmitter();
        server.bind = (options) => bindings.push(options);
        server.close = () => {{}};
        return server;
      }},
    }};
  }}
  if (request === "max-api") {{
    return {{
      outlet: () => Promise.resolve(),
      post: () => {{}},
      POST_LEVELS: {{ ERROR: "error", WARN: "warn" }},
    }};
  }}
  return originalLoad.call(this, request, parent, isMain);
}};

process.env.SCRIPT_PATH = receiverPath;
import(pathToFileURL(receiverPath).href).then(() => {{
  process.stdout.write(JSON.stringify(bindings));
}});
"""
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            [{"address": "127.0.0.1", "port": 9000, "exclusive": True}],
        )

    def test_node_receiver_socket_binds_only_to_loopback(self) -> None:
        script = f"""
const receiver = require({json.dumps(str(RECEIVER_JS_PATH))});
const server = receiver.createLoopbackReceiver({{ port: 0 }});
server.once("listening", () => {{
  const address = server.address();
  server.close(() => process.stdout.write(JSON.stringify(address)));
}});
"""
        completed = subprocess.run(
            ["node", "-e", script],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        address = json.loads(completed.stdout)

        self.assertEqual(address["address"], "127.0.0.1")
        self.assertEqual(address["family"], "IPv4")
        self.assertGreater(address["port"], 0)


if __name__ == "__main__":
    unittest.main()
