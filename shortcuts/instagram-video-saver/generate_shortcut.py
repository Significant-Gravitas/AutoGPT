#!/usr/bin/env python3
"""
generate_shortcut.py — Claude-InstaSaver v1.0
Generates Claude-InstaSaver.shortcut (binary plist) and shortcut-source.json
from a Python-defined workflow.

Usage:
    python3 generate_shortcut.py

Output:
    Claude-InstaSaver.shortcut   — importable directly into iOS Shortcuts app
    shortcut-source.json         — human-readable workflow definition

Requirements:
    Python 3.6+ (uses stdlib only: plistlib, json, uuid)
"""

import json
import plistlib
import uuid
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helper: build a WFWorkflowAction dict
# ---------------------------------------------------------------------------

def action(identifier: str, **params) -> dict:
    """Return a single Shortcuts action dict."""
    d = {"WFWorkflowActionIdentifier": identifier}
    if params:
        d["WFWorkflowActionParameters"] = params
    return d


def variable_ref(name: str, output_name: str = None) -> dict:
    """Return a Variable reference (Magic Variable)."""
    ref = {
        "Type": "Variable",
        "VariableName": name,
    }
    if output_name:
        ref["OutputName"] = output_name
    return {"Value": ref, "WFSerializationType": "WFTextTokenAttachment"}


def text_token(text: str) -> dict:
    """Return a WFTextTokenString value (plain text that may embed variables)."""
    return {
        "Value": {
            "attachmentsByRange": {},
            "string": text,
        },
        "WFSerializationType": "WFTextTokenString",
    }


def input_ref(name: str = "Shortcut Input") -> dict:
    """Reference the shortcut's input variable."""
    return {
        "Value": {
            "Type": "Variable",
            "VariableName": name,
        },
        "WFSerializationType": "WFTextTokenAttachment",
    }


def magic_variable(action_output_uuid: str, output_name: str) -> dict:
    """Reference the output of a named action (magic variable)."""
    return {
        "Value": {
            "OutputUUID": action_output_uuid,
            "Type": "ActionOutput",
            "OutputName": output_name,
        },
        "WFSerializationType": "WFTextTokenAttachment",
    }


# ---------------------------------------------------------------------------
# Build the WFWorkflowActions list
# ---------------------------------------------------------------------------
# Each action in the list is a dict.  UUIDs are generated per-action so that
# magic variable references work correctly.
# ---------------------------------------------------------------------------

# Pre-generate UUIDs for actions whose outputs we reference later.
UUID_INPUT_URL   = str(uuid.uuid4()).upper()
UUID_ASKED_URL   = str(uuid.uuid4()).upper()
UUID_SHORTCODE   = str(uuid.uuid4()).upper()
UUID_API_URL     = str(uuid.uuid4()).upper()
UUID_API_RESP    = str(uuid.uuid4()).upper()
UUID_VIDEO_URL_A = str(uuid.uuid4()).upper()
UUID_INSTA_URL   = str(uuid.uuid4()).upper()
UUID_ENCODED_URL = str(uuid.uuid4()).upper()
UUID_FB_RESP     = str(uuid.uuid4()).upper()
UUID_VIDEO_URL_B = str(uuid.uuid4()).upper()
UUID_FINAL_URL   = str(uuid.uuid4()).upper()
UUID_VIDEO_DATA  = str(uuid.uuid4()).upper()

# Group UUIDs for if/else blocks (must be matching pairs)
UUID_IF_NO_INPUT   = str(uuid.uuid4()).upper()
UUID_IF_BAD_URL    = str(uuid.uuid4()).upper()
UUID_IF_NO_VIDEO_A = str(uuid.uuid4()).upper()
UUID_IF_NO_VIDEO_B = str(uuid.uuid4()).upper()

# ---------------------------------------------------------------------------

ACTIONS = [

    # ── 0. Header comment ────────────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.comment",
        "WFWorkflowActionParameters": {
            "WFCommentActionText": (
                "Claude-InstaSaver v1.0\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Downloads Instagram videos and saves them to your Photos library.\n\n"
                "Usage:\n"
                "  • Share an Instagram post/reel URL via the Share Sheet, OR\n"
                "  • Open the shortcut directly and paste a URL when prompted.\n\n"
                "Works with: /p/ posts, /reel/ reels, /tv/ IGTV videos.\n"
                "Note: subject to Instagram ToS — for personal use only.\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        },
    },

    # ── 1. Get Shortcut Input (URL from share sheet) ──────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.shortcut.input",
        "WFWorkflowActionParameters": {
            "UUID": UUID_INPUT_URL,
            "CustomOutputName": "Shortcut Input",
        },
    },

    # ── 2. If there is NO input, ask the user for a URL ──────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": UUID_IF_NO_INPUT,
            "GroupingIdentifier": UUID_IF_NO_INPUT,
            "WFControlFlowMode": 0,          # if
            "WFCondition": 101,              # "has no value"
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Shortcut Input",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
        },
    },

    # ── 3.   Ask for URL ─────────────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.ask",
        "WFWorkflowActionParameters": {
            "UUID": UUID_ASKED_URL,
            "CustomOutputName": "Asked URL",
            "WFAskActionPrompt": "Paste an Instagram video URL:",
            "WFInputType": "URL",
            "WFAskActionDefaultAnswer": "https://www.instagram.com/reel/",
        },
    },

    # ── 4.   Set "Shortcut Input" to the asked URL ────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.setvariable",
        "WFWorkflowActionParameters": {
            "WFVariableName": "Shortcut Input",
            "WFInput": magic_variable(UUID_ASKED_URL, "Asked URL"),
        },
    },

    # ── 5. End if (no input) ──────────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": str(uuid.uuid4()).upper(),
            "GroupingIdentifier": UUID_IF_NO_INPUT,
            "WFControlFlowMode": 2,          # end if
        },
    },

    # ── 6. Validate: URL must contain "instagram.com" ────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": UUID_IF_BAD_URL,
            "GroupingIdentifier": UUID_IF_BAD_URL,
            "WFControlFlowMode": 0,          # if
            "WFCondition": 999,              # "does not contain"
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Shortcut Input",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
            "WFConditionalActionString": "instagram.com",
        },
    },

    # ── 7.   Alert — invalid URL ─────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.showresult",
        "WFWorkflowActionParameters": {
            "Text": text_token(
                "❌ Invalid URL\n\n"
                "Please share a valid Instagram post, reel, or IGTV URL.\n"
                "Example: https://www.instagram.com/reel/ABC123/"
            ),
        },
    },

    # ── 8. End if (bad URL) — shortcut terminates here for bad URLs ───────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": str(uuid.uuid4()).upper(),
            "GroupingIdentifier": UUID_IF_BAD_URL,
            "WFControlFlowMode": 2,          # end if
        },
    },

    # ── 9. Match Regex — extract shortcode from URL ───────────────────────────
    #   Matches: /p/CODE, /reel/CODE, /tv/CODE
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.text.match",
        "WFWorkflowActionParameters": {
            "UUID": UUID_SHORTCODE,
            "CustomOutputName": "Shortcode Match",
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Shortcut Input",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
            "WFMatchTextPattern": r"instagram\.com/(?:p|reel|tv)/([A-Za-z0-9_-]+)",
            "WFMatchTextCaseSensitive": False,
        },
    },

    # ── 10. Get first capture group (index 1) → shortcode string ─────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.getitemfromlist",
        "WFWorkflowActionParameters": {
            "WFInput": magic_variable(UUID_SHORTCODE, "Shortcode Match"),
            "WFItemSpecifier": "Item At Index",
            "WFItemIndex": 1,
            "UUID": str(uuid.uuid4()).upper(),
            "CustomOutputName": "Shortcode",
        },
    },

    # ── 11. Set variable "shortcode" ──────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.setvariable",
        "WFWorkflowActionParameters": {
            "WFVariableName": "shortcode",
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Shortcode",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
        },
    },

    # ── 12. Build Instagram API URL ───────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.url",
        "WFWorkflowActionParameters": {
            "UUID": UUID_API_URL,
            "CustomOutputName": "API URL",
            "WFURLActionURL": {
                "Value": {
                    "attachmentsByRange": {
                        "{35, 1}": {
                            "Type": "Variable",
                            "VariableName": "shortcode",
                        }
                    },
                    "string": "https://www.instagram.com/p/%@/?__a=1&__d=dis",
                },
                "WFSerializationType": "WFTextTokenString",
            },
        },
    },

    # ── 13. GET Instagram internal API ───────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
        "WFWorkflowActionParameters": {
            "UUID": UUID_API_RESP,
            "CustomOutputName": "API Response",
            "WFURL": magic_variable(UUID_API_URL, "API URL"),
            "WFHTTPMethod": "GET",
            "WFHTTPHeaders": {
                "Value": {
                    "WFDictionaryFieldValueItems": [
                        {
                            "WFItemType": 0,
                            "WFValue": {
                                "Value": {
                                    "attachmentsByRange": {},
                                    "string": (
                                        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
                                        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
                                        "Mobile/15E148 Safari/604.1"
                                    ),
                                },
                                "WFSerializationType": "WFTextTokenString",
                            },
                            "WFKey": {
                                "Value": {
                                    "attachmentsByRange": {},
                                    "string": "User-Agent",
                                },
                                "WFSerializationType": "WFTextTokenString",
                            },
                        },
                        {
                            "WFItemType": 0,
                            "WFValue": {
                                "Value": {
                                    "attachmentsByRange": {},
                                    "string": "https://www.instagram.com/",
                                },
                                "WFSerializationType": "WFTextTokenString",
                            },
                            "WFKey": {
                                "Value": {
                                    "attachmentsByRange": {},
                                    "string": "Referer",
                                },
                                "WFSerializationType": "WFTextTokenString",
                            },
                        },
                    ]
                },
                "WFSerializationType": "WFDictionaryFieldValue",
            },
            "WFShowWebView": False,
        },
    },

    # ── 14. Parse JSON response → dictionary ──────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.getdictionaryvalue",
        "WFWorkflowActionParameters": {
            "UUID": UUID_VIDEO_URL_A,
            "CustomOutputName": "Video URL (Method A)",
            "WFInput": magic_variable(UUID_API_RESP, "API Response"),
            "WFKey": {
                "Value": {
                    "attachmentsByRange": {},
                    "string": "items.0.video_url",
                },
                "WFSerializationType": "WFTextTokenString",
            },
        },
    },

    # ── 15. If Method A produced a video URL → use it; else try fallback ──────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": UUID_IF_NO_VIDEO_A,
            "GroupingIdentifier": UUID_IF_NO_VIDEO_A,
            "WFControlFlowMode": 0,          # if
            "WFCondition": 101,              # has no value
            "WFInput": magic_variable(UUID_VIDEO_URL_A, "Video URL (Method A)"),
        },
    },

    # ── 16.   FALLBACK: URL-encode the original Instagram URL ─────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.urlencode",
        "WFWorkflowActionParameters": {
            "UUID": UUID_ENCODED_URL,
            "CustomOutputName": "Encoded URL",
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Shortcut Input",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
        },
    },

    # ── 17.   Build fallback service URL ─────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.url",
        "WFWorkflowActionParameters": {
            "UUID": UUID_INSTA_URL,
            "CustomOutputName": "Fallback API URL",
            "WFURLActionURL": {
                "Value": {
                    "attachmentsByRange": {
                        "{38, 1}": {
                            "OutputUUID": UUID_ENCODED_URL,
                            "Type": "ActionOutput",
                            "OutputName": "Encoded URL",
                        }
                    },
                    "string": "https://sssinstagram.com/api/convert?url=%@",
                },
                "WFSerializationType": "WFTextTokenString",
            },
        },
    },

    # ── 18.   GET fallback service ────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
        "WFWorkflowActionParameters": {
            "UUID": UUID_FB_RESP,
            "CustomOutputName": "Fallback Response",
            "WFURL": magic_variable(UUID_INSTA_URL, "Fallback API URL"),
            "WFHTTPMethod": "GET",
            "WFShowWebView": False,
        },
    },

    # ── 19.   Extract video URL from fallback JSON ────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.getdictionaryvalue",
        "WFWorkflowActionParameters": {
            "UUID": UUID_VIDEO_URL_B,
            "CustomOutputName": "Video URL (Method B)",
            "WFInput": magic_variable(UUID_FB_RESP, "Fallback Response"),
            "WFKey": {
                "Value": {
                    "attachmentsByRange": {},
                    "string": "download_url",
                },
                "WFSerializationType": "WFTextTokenString",
            },
        },
    },

    # ── 20.   Set "Final Video URL" from fallback ─────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.setvariable",
        "WFWorkflowActionParameters": {
            "WFVariableName": "Final Video URL",
            "WFInput": magic_variable(UUID_VIDEO_URL_B, "Video URL (Method B)"),
        },
    },

    # ── 21. Else: use Method A result ─────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": str(uuid.uuid4()).upper(),
            "GroupingIdentifier": UUID_IF_NO_VIDEO_A,
            "WFControlFlowMode": 1,          # else
        },
    },

    # ── 22.   Set "Final Video URL" from Method A ─────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.setvariable",
        "WFWorkflowActionParameters": {
            "WFVariableName": "Final Video URL",
            "WFInput": magic_variable(UUID_VIDEO_URL_A, "Video URL (Method A)"),
        },
    },

    # ── 23. End if (method A/B) ───────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": str(uuid.uuid4()).upper(),
            "GroupingIdentifier": UUID_IF_NO_VIDEO_A,
            "WFControlFlowMode": 2,          # end if
        },
    },

    # ── 24. Validate we have a Final Video URL ────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": UUID_IF_NO_VIDEO_B,
            "GroupingIdentifier": UUID_IF_NO_VIDEO_B,
            "WFControlFlowMode": 0,          # if
            "WFCondition": 101,              # has no value
            "WFInput": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Final Video URL",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
        },
    },

    # ── 25.   Error: could not extract video ──────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.showresult",
        "WFWorkflowActionParameters": {
            "Text": text_token(
                "❌ Could Not Download Video\n\n"
                "Both download methods failed. This may be because:\n"
                "• The post is from a private account you don't follow\n"
                "• Instagram has changed its internal API\n"
                "• The fallback service is temporarily unavailable\n"
                "• The post does not contain a video\n\n"
                "See ISSUES.md in the Claude-InstaSaver repo for workarounds."
            ),
        },
    },

    # ── 26. End if (no video URL) ─────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "UUID": str(uuid.uuid4()).upper(),
            "GroupingIdentifier": UUID_IF_NO_VIDEO_B,
            "WFControlFlowMode": 2,          # end if
        },
    },

    # ── 27. Download the video file ───────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
        "WFWorkflowActionParameters": {
            "UUID": UUID_VIDEO_DATA,
            "CustomOutputName": "Video File",
            "WFURL": {
                "Value": {
                    "Type": "Variable",
                    "VariableName": "Final Video URL",
                },
                "WFSerializationType": "WFTextTokenAttachment",
            },
            "WFHTTPMethod": "GET",
            "WFShowWebView": False,
        },
    },

    # ── 28. Save to Photos (Camera Roll) ─────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.savephotosalbum",
        "WFWorkflowActionParameters": {
            "WFInput": magic_variable(UUID_VIDEO_DATA, "Video File"),
            "WFAlbumName": "Recents",
        },
    },

    # ── 29. Success notification ──────────────────────────────────────────────
    {
        "WFWorkflowActionIdentifier": "is.workflow.actions.notification",
        "WFWorkflowActionParameters": {
            "WFNotificationActionTitle": "Claude-InstaSaver",
            "WFNotificationActionBody": "✅ Instagram video saved to Photos!",
            "WFNotificationActionSound": True,
        },
    },
]

# ---------------------------------------------------------------------------
# Shortcut metadata
# ---------------------------------------------------------------------------

SHORTCUT = {
    # Client version string from a recent Shortcuts app release
    "WFWorkflowClientVersion": "2605.0.5",
    # 900 = iOS 15 minimum (the lowest we want to support)
    "WFWorkflowMinimumClientVersion": 900,
    "WFWorkflowMinimumClientVersionString": "900",
    "WFWorkflowName": "Claude-InstaSaver",
    "WFWorkflowHasShortcutInputVariables": True,
    # Camera icon (camera-fill glyph) with Instagram-gradient-ish color
    "WFWorkflowIcon": {
        "WFWorkflowIconStartColor": -1618153729,  # purple-ish
        "WFWorkflowIconGlyphNumber": 59511,        # camera glyph
    },
    # Accept URLs and strings from the share sheet
    "WFWorkflowInputContentItemClasses": [
        "WFURLContentItem",
        "WFStringContentItem",
    ],
    # ActionExtension = appears in Share Sheet; WatchKit removed for simplicity
    "WFWorkflowTypes": ["ActionExtension"],
    "WFWorkflowActions": ACTIONS,
}

# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------

def write_shortcut():
    out_path = OUTPUT_DIR / "Claude-InstaSaver.shortcut"
    with open(out_path, "wb") as f:
        plistlib.dump(SHORTCUT, f, fmt=plistlib.FMT_BINARY)
    size = out_path.stat().st_size
    print(f"[OK] {out_path}  ({size:,} bytes)")
    return size


def write_json_source():
    out_path = OUTPUT_DIR / "shortcut-source.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(SHORTCUT, f, indent=2, ensure_ascii=False, default=str)
    size = out_path.stat().st_size
    print(f"[OK] {out_path}  ({size:,} bytes)")
    return size


if __name__ == "__main__":
    print("Generating Claude-InstaSaver shortcut files …\n")
    sc_size = write_shortcut()
    js_size = write_json_source()
    print(f"\nDone. Shortcut contains {len(ACTIONS)} actions.")
    print("Import Claude-InstaSaver.shortcut into the iOS Shortcuts app")
    print("via AirDrop, Files app, or iCloud Drive.")
