import { describe, expect, it } from "vitest";
import { classifyArtifact } from "./helpers";

describe("classifyArtifact", () => {
  it("prefers explicit code extensions over markdown mime types", () => {
    expect(classifyArtifact("text/markdown", "main.py").type).toBe("code");
  });

  it("treats jsx and tsx files as React previews", () => {
    expect(classifyArtifact("text/plain", "Widget.jsx").type).toBe("react");
    expect(classifyArtifact("text/plain", "Widget.tsx").type).toBe("react");
  });

  it("keeps pdf files previewable even when the mime type is octet-stream", () => {
    expect(
      classifyArtifact("application/octet-stream", "report.pdf").type,
    ).toBe("pdf");
  });
});
