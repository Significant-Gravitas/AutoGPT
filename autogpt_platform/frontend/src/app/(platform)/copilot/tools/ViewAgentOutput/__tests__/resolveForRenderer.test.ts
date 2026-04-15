import { describe, expect, it } from "vitest";
import { resolveForRenderer } from "../ViewAgentOutput";
import { globalRegistry } from "@/components/contextual/OutputRenderers";

describe("resolveForRenderer", () => {
  it("preserves workspace image URI for the registry to handle", () => {
    const result = resolveForRenderer("workspace://abc123#image/png");
    expect(String(result.value)).toMatch(/^workspace:\/\//);
    expect(result.metadata?.mimeType).toBe("image/png");
  });

  it("preserves workspace video URI for the registry to handle", () => {
    const result = resolveForRenderer("workspace://vid456#video/mp4");
    expect(String(result.value)).toMatch(/^workspace:\/\//);
    expect(result.metadata?.mimeType).toBe("video/mp4");
  });

  it("passes non-workspace values through unchanged", () => {
    const result = resolveForRenderer("just a string");
    expect(result.value).toBe("just a string");
    expect(result.metadata).toBeUndefined();
  });

  it("passes non-string values through unchanged", () => {
    const obj = { foo: "bar" };
    const result = resolveForRenderer(obj);
    expect(result.value).toBe(obj);
    expect(result.metadata).toBeUndefined();
  });

  it("workspace image URIs match WorkspaceFileRenderer with loading/error states", () => {
    // WorkspaceFileRenderer (priority 50) should handle workspace:// URIs
    // since resolveForRenderer no longer pre-converts them to proxy URLs.
    const resolved = resolveForRenderer("workspace://abc123#image/png");
    const renderer = globalRegistry.getRenderer(
      resolved.value,
      resolved.metadata,
    );
    expect(renderer).toBeDefined();
    expect(renderer!.name).toBe("WorkspaceFileRenderer");
  });

  it("workspace video URIs match WorkspaceFileRenderer", () => {
    const resolved = resolveForRenderer("workspace://vid456#video/mp4");
    const renderer = globalRegistry.getRenderer(
      resolved.value,
      resolved.metadata,
    );
    expect(renderer).toBeDefined();
    expect(renderer!.name).toBe("WorkspaceFileRenderer");
  });
});
