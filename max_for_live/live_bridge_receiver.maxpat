{
  "patcher": {
    "fileversion": 1,
    "appversion": {
      "major": 8,
      "minor": 6,
      "revision": 0,
      "architecture": "x64",
      "modernui": 1
    },
    "classnamespace": "box",
    "rect": [34.0, 79.0, 900.0, 360.0],
    "bglocked": 0,
    "openinpresentation": 0,
    "default_fontsize": 12.0,
    "default_fontface": 0,
    "default_fontname": "Arial",
    "gridonopen": 1,
    "gridsize": [15.0, 15.0],
    "gridsnaponopen": 1,
    "toolbarvisible": 1,
    "boxes": [
      {
        "box": {
          "id": "obj-1",
          "maxclass": "comment",
          "text": "Codex Live Bridge: receive UDP JSON on port 9000 and execute via LiveAPI router JS",
          "patching_rect": [24.0, 16.0, 660.0, 20.0]
        }
      },
      {
        "box": {
          "id": "obj-2",
          "maxclass": "newobj",
          "text": "udpreceive 9000 @outputformat rawbytes @defer 1",
          "patching_rect": [24.0, 52.0, 280.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-3",
          "maxclass": "newobj",
          "text": "js /Users/michaelwall/codex-live-bridge/max_for_live/live_api_command_router.js",
          "patching_rect": [24.0, 90.0, 570.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-4",
          "maxclass": "newobj",
          "text": "print live_bridge_router",
          "patching_rect": [24.0, 128.0, 150.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-6",
          "maxclass": "newobj",
          "text": "udpsend 127.0.0.1 9002",
          "patching_rect": [196.0, 128.0, 170.0, 22.0]
        }
      },
      {
        "box": {
          "id": "obj-5",
          "maxclass": "comment",
          "text": "Copy these objects into your Max MIDI Effect patcher (Edit in Max), then save the device. Keep udpsend for query responses.",
          "patching_rect": [24.0, 166.0, 620.0, 20.0]
        }
      }
    ],
    "lines": [
      {
        "patchline": {
          "source": ["obj-2", 0],
          "destination": ["obj-3", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-3", 0],
          "destination": ["obj-4", 0]
        }
      },
      {
        "patchline": {
          "source": ["obj-3", 0],
          "destination": ["obj-6", 0]
        }
      }
    ]
  }
}
