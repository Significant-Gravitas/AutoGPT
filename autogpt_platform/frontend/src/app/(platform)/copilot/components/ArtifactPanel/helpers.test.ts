import { describe, expect, it } from "vitest";
import { classifyArtifact } from "./helpers";

describe("classifyArtifact", () => {
  it("routes PDF by extension", () => {
    const c = classifyArtifact(null, "report.pdf");
    expect(c.type).toBe("pdf");
    expect(c.openable).toBe(true);
  });

  it("routes PDF by MIME when no extension matches", () => {
    const c = classifyArtifact("application/pdf", "noextension");
    expect(c.type).toBe("pdf");
  });

  it("routes JSX/TSX as react", () => {
    expect(classifyArtifact(null, "App.tsx").type).toBe("react");
    expect(classifyArtifact(null, "Comp.jsx").type).toBe("react");
  });

  it("routes code extensions to code", () => {
    expect(classifyArtifact(null, "script.py").type).toBe("code");
    expect(classifyArtifact(null, "main.go").type).toBe("code");
    expect(classifyArtifact(null, "Dockerfile.yml").type).toBe("code");
  });

  it("treats images as image (inline rendered)", () => {
    expect(classifyArtifact(null, "photo.png").type).toBe("image");
    expect(classifyArtifact("image/svg+xml", "unknown").type).toBe("image");
  });

  it("treats CSVs as csv with source toggle", () => {
    const c = classifyArtifact(null, "data.csv");
    expect(c.type).toBe("csv");
    expect(c.hasSourceToggle).toBe(true);
  });

  it("treats HTML as html with source toggle", () => {
    expect(classifyArtifact(null, "page.html").type).toBe("html");
    expect(classifyArtifact("text/html", "noext").type).toBe("html");
  });

  it("treats markdown as markdown", () => {
    expect(classifyArtifact(null, "README.md").type).toBe("markdown");
    expect(classifyArtifact("text/markdown", "x").type).toBe("markdown");
  });

  it("gates files > 10MB to download-only", () => {
    const c = classifyArtifact("text/plain", "big.txt", 20 * 1024 * 1024);
    expect(c.openable).toBe(false);
    expect(c.type).toBe("download-only");
  });

  it("treats binary/octet-stream MIME as download-only", () => {
    expect(classifyArtifact("application/zip", "a.zip").openable).toBe(false);
    expect(classifyArtifact("application/octet-stream", "x").openable).toBe(
      false,
    );
    expect(classifyArtifact("video/mp4", "clip.mp4").openable).toBe(false);
  });

  it("defaults unknown extension+MIME to download-only (not text)", () => {
    // Regression: previously dumped binary as <pre>; now refuses to open.
    const c = classifyArtifact(null, "data.bin");
    expect(c.openable).toBe(false);
    expect(c.type).toBe("download-only");
  });

  it("is case-insensitive on extension", () => {
    expect(classifyArtifact(null, "image.PNG").type).toBe("image");
    expect(classifyArtifact(null, "Notes.MD").type).toBe("markdown");
  });

  it("prioritizes extension over MIME", () => {
    // Extension says CSV, MIME says plain text → extension wins.
    const c = classifyArtifact("text/plain", "data.csv");
    expect(c.type).toBe("csv");
  });
});
