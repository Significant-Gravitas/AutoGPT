import { describe, expect, test, vi } from "vitest";

// react-force-graph-2d uses HTMLCanvas + window APIs at import time, so
// stub it out — these tests only exercise GraphCanvas's exported pure
// helpers, not the canvas itself.
vi.mock("react-force-graph-2d", () => ({
  default: () => null,
}));

import {
  escapeHtml,
  getLinkTooltipLabel,
  getNodeTooltipLabel,
  linkTouchesNode,
} from "../GraphCanvas";

describe("tooltip labels stay inert through float-tooltip's innerHTML sink", () => {
  test("node tooltip renders a malicious memory-derived name as text, not markup", () => {
    const payload = '<img src=x onerror="window.__pwned = true">';
    const label = getNodeTooltipLabel({
      uuid: "node-1",
      label: "Entity",
      type: "Person",
      name: payload,
    });

    // float-tooltip assigns string labels via innerHTML — simulate that
    // sink and assert the name cannot create elements.
    const host = document.createElement("div");
    host.innerHTML = label;
    expect(host.querySelector("img")).toBeNull();
    expect(host.textContent).toBe(`Person: ${payload}`);
  });

  test("edge tooltip renders a malicious edge name as text, not markup", () => {
    const payload = "<script>document.title = 'pwned'</script>";
    const label = getLinkTooltipLabel({
      uuid: "edge-1",
      label: "RELATES_TO",
      source: "a",
      target: "b",
      name: payload,
    });

    const host = document.createElement("div");
    host.innerHTML = label;
    expect(host.querySelector("script")).toBeNull();
    expect(host.textContent).toBe(`RELATES_TO: ${payload}`);
  });

  test("edge tooltip falls back to the relationship label when unnamed", () => {
    const label = getLinkTooltipLabel({
      uuid: "edge-2",
      label: "MENTIONS",
      source: "a",
      target: "b",
    });
    expect(label).toBe("MENTIONS");
  });

  test("node tooltip falls back to the uuid prefix when unnamed", () => {
    const label = getNodeTooltipLabel({
      uuid: "abcdef1234567890",
      label: "Entity",
    });
    expect(label).toBe("Entity: abcdef12");
  });

  test("escapeHtml escapes every HTML-significant character", () => {
    expect(escapeHtml(`&<>"'`)).toBe("&amp;&lt;&gt;&quot;&#39;");
  });
});

describe("selected-node edge highlighting handles both link endpoint shapes", () => {
  const nodeA = { uuid: "node-a", label: "Entity" };
  const nodeB = { uuid: "node-b", label: "Entity" };

  test("matches string endpoints before d3 swaps in node objects", () => {
    const link = {
      source: "node-a",
      target: "node-b",
      uuid: "e1",
      label: "RELATES_TO",
    };
    expect(linkTouchesNode(link, "node-a")).toBe(true);
    expect(linkTouchesNode(link, "node-b")).toBe(true);
    expect(linkTouchesNode(link, "node-c")).toBe(false);
  });

  test("matches object endpoints after d3 replaces ids with node refs", () => {
    const link = {
      source: nodeA,
      target: nodeB,
      uuid: "e1",
      label: "RELATES_TO",
    };
    expect(linkTouchesNode(link, "node-a")).toBe(true);
    expect(linkTouchesNode(link, "node-b")).toBe(true);
    expect(linkTouchesNode(link, "node-c")).toBe(false);
  });

  test("highlights nothing when no node is selected", () => {
    const link = {
      source: nodeA,
      target: "node-b",
      uuid: "e1",
      label: "RELATES_TO",
    };
    expect(linkTouchesNode(link, null)).toBe(false);
  });
});
