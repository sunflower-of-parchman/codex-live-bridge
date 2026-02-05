import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = REPO_ROOT / "max_for_live" / "live_bridge_receiver.maxpat"


class MaxForLivePatchTests(unittest.TestCase):
    def test_patch_file_exists(self):
        self.assertTrue(PATCH_PATH.exists())

    def test_patch_contains_udp_receiver_and_js_router(self):
        patch_data = json.loads(PATCH_PATH.read_text(encoding="utf-8"))
        boxes = patch_data["patcher"]["boxes"]
        texts = [box["box"].get("text", "") for box in boxes]
        self.assertTrue(any("udpreceive 9000" in text for text in texts))
        self.assertTrue(any("live_api_command_router.js" in text for text in texts))

    def test_patch_routes_js_output_to_print(self):
        patch_data = json.loads(PATCH_PATH.read_text(encoding="utf-8"))
        lines = patch_data["patcher"]["lines"]
        route_exists = any(
            line["patchline"].get("source") == ["obj-3", 0]
            and line["patchline"].get("destination") == ["obj-4", 0]
            for line in lines
        )
        self.assertTrue(route_exists, "Expected js outlet to be wired to print live_bridge_router.")


if __name__ == "__main__":
    unittest.main()
