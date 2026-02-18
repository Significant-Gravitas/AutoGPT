import { describe, expect, test } from "vitest";
import { RJSFSchema } from "@rjsf/utils";
import { buildVisibleOutputTree } from "./output-visibility";

const outputProperties: Record<string, RJSFSchema> = {
  main_result: {
    type: "object",
    title: "Main Result",
    properties: {
      text: {
        type: "string",
        title: "Text",
      },
      metadata: {
        type: "object",
        title: "Metadata",
        properties: {
          score: {
            type: "number",
            title: "Score",
          },
          debug: {
            type: "string",
            title: "Debug",
          },
        },
      },
    },
  },
  status: {
    type: "string",
    title: "Status",
  },
};

function getVisibleHandles(
  connectedHandles: Set<string>,
  collapsedHandles: Record<string, boolean> = {},
) {
  const tree = buildVisibleOutputTree({
    properties: outputProperties,
    isHandleConnected: (handleId) => connectedHandles.has(handleId),
    isCollapsed: (handleId) => collapsedHandles[handleId] ?? true,
  });

  return flattenHandles(tree);
}

function flattenHandles(
  nodes: ReturnType<typeof buildVisibleOutputTree>,
): string[] {
  const handles: string[] = [];

  function walk(currentNodes: ReturnType<typeof buildVisibleOutputTree>) {
    currentNodes.forEach((node) => {
      handles.push(node.fullKey);
      walk(node.children);
    });
  }

  walk(nodes);
  return handles;
}

describe("buildVisibleOutputTree", () => {
  test("keeps disconnected top-level outputs visible", () => {
    const visibleHandles = getVisibleHandles(new Set());

    expect(visibleHandles).toContain("main_result");
    expect(visibleHandles).toContain("status");
    expect(visibleHandles).not.toContain("main_result_#_text");
  });

  test("keeps connected sub-output visible when parent object is collapsed", () => {
    const visibleHandles = getVisibleHandles(new Set(["main_result_#_text"]));

    expect(visibleHandles).toContain("main_result");
    expect(visibleHandles).toContain("main_result_#_text");
  });

  test("keeps object ancestor visible when a deep descendant is connected", () => {
    const visibleHandles = getVisibleHandles(
      new Set(["main_result_#_metadata_#_score"]),
    );

    expect(visibleHandles).toContain("main_result");
    expect(visibleHandles).toContain("main_result_#_metadata");
    expect(visibleHandles).toContain("main_result_#_metadata_#_score");
    expect(visibleHandles).not.toContain("main_result_#_text");
  });

  test("shows disconnected children when object output is expanded", () => {
    const visibleHandles = getVisibleHandles(new Set(), { main_result: false });

    expect(visibleHandles).toContain("main_result_#_text");
    expect(visibleHandles).toContain("main_result_#_metadata");
  });

  test("shows only connected nested branch when parents are collapsed", () => {
    const visibleHandles = getVisibleHandles(
      new Set(["main_result_#_metadata_#_score"]),
      {
        main_result: true,
        "main_result_#_metadata": true,
      },
    );

    expect(visibleHandles).toContain("main_result_#_metadata_#_score");
    expect(visibleHandles).not.toContain("main_result_#_metadata_#_debug");
  });
});
