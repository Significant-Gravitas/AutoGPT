import { describe, expect, test } from "vitest";
import { selectRecordingSettings, toCapturedStep } from "../recording-helpers";

describe("selectRecordingSettings", () => {
  test("uses the default structured route and every advertised channel", () => {
    expect(
      selectRecordingSettings(
        ["screenshots_to_cloud", "local_vlm", "extract_then_cloud"],
        ["desktop_ax", "browser", "floor"],
      ),
    ).toEqual({
      interpretationRoute: "extract_then_cloud",
      channels: ["floor", "browser", "desktop_ax"],
    });
  });

  test("prefers a local model over the screenshot fallback", () => {
    expect(
      selectRecordingSettings(["screenshots_to_cloud", "local_vlm"], ["floor"]),
    ).toEqual({
      interpretationRoute: "local_vlm",
      channels: ["floor"],
    });
  });

  test("requires both an advertised route and channel", () => {
    expect(selectRecordingSettings([], ["floor"])).toBeNull();
    expect(selectRecordingSettings(["extract_then_cloud"], [])).toBeNull();
  });
});

test("toCapturedStep preserves every retained trajectory field", () => {
  expect(
    toCapturedStep({
      seq: 7,
      ts: 123.5,
      actor: "human",
      action: "fill",
      screenshot_ref: "frame-7",
      cursor: [120, 240],
      active_app: "Chrome",
      active_window: "Customer form",
      narration: "Enter the customer name",
      enrichment: {
        kind: "dom",
        selectors: [{ strategy: "css", value: "#customer-name" }],
        ax_path: "Application/Window/TextField",
        role: "textbox",
        label: "Customer name",
        url: "https://example.test/customers/new",
      },
      value: {
        raw: { first: "Ada" },
        type: "text",
        is_parameter: true,
      },
      outcome: "ok",
      redacted: false,
    }),
  ).toEqual({
    seq: 7,
    timestamp: 123.5,
    actor: "human",
    action: "fill",
    label: "Customer name",
    screenshotRef: "frame-7",
    cursor: [120, 240],
    activeApp: "Chrome",
    activeWindow: "Customer form",
    enrichment: {
      kind: "dom",
      selectors: [{ strategy: "css", value: "#customer-name" }],
      axPath: "Application/Window/TextField",
      role: "textbox",
      label: "Customer name",
      url: "https://example.test/customers/new",
    },
    narration: "Enter the customer name",
    value: '{"first":"Ada"}',
    valueType: "text",
    isParameter: true,
    outcome: "ok",
    redacted: false,
  });
});
