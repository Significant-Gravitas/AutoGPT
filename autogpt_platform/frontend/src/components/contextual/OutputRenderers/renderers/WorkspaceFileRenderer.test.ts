import { describe, expect, it } from "vitest";
import { parseWorkspaceURI } from "./WorkspaceFileRenderer";
import { parseWorkspaceFileID } from "@/components/renderers/InputRenderer/base/standard/widgets/FileInput/useWorkspaceUpload";

describe("parseWorkspaceURI", () => {
  it("parses a full workspace URI with mime type", () => {
    const result = parseWorkspaceURI("workspace://file-abc-123#image/png");
    expect(result).toEqual({ fileID: "file-abc-123", mimeType: "image/png" });
  });

  it("parses a workspace URI without mime type", () => {
    const result = parseWorkspaceURI("workspace://file-abc-123");
    expect(result).toEqual({ fileID: "file-abc-123", mimeType: null });
  });

  it("returns null for non-workspace URIs", () => {
    expect(parseWorkspaceURI("https://example.com")).toBeNull();
    expect(parseWorkspaceURI("data:image/png;base64,abc")).toBeNull();
    expect(parseWorkspaceURI("")).toBeNull();
    expect(parseWorkspaceURI("file:///tmp/test.txt")).toBeNull();
  });

  it("handles empty fragment after hash as null mime type", () => {
    const result = parseWorkspaceURI("workspace://file-abc-123#");
    expect(result).toEqual({ fileID: "file-abc-123", mimeType: null });
  });

  it("handles mime types with subtype", () => {
    const result = parseWorkspaceURI(
      "workspace://file-id#application/octet-stream",
    );
    expect(result).toEqual({
      fileID: "file-id",
      mimeType: "application/octet-stream",
    });
  });

  it("handles UUID-style file IDs", () => {
    const uuid = "550e8400-e29b-41d4-a716-446655440000";
    const result = parseWorkspaceURI(`workspace://${uuid}#text/plain`);
    expect(result).toEqual({ fileID: uuid, mimeType: "text/plain" });
  });
});

describe("parseWorkspaceFileID", () => {
  it("extracts file ID from a full workspace URI", () => {
    expect(parseWorkspaceFileID("workspace://file-abc-123#image/png")).toBe(
      "file-abc-123",
    );
  });

  it("extracts file ID when no mime type fragment", () => {
    expect(parseWorkspaceFileID("workspace://file-abc-123")).toBe(
      "file-abc-123",
    );
  });

  it("returns null for non-workspace URIs", () => {
    expect(parseWorkspaceFileID("https://example.com")).toBeNull();
    expect(parseWorkspaceFileID("data:image/png;base64,abc")).toBeNull();
    expect(parseWorkspaceFileID("")).toBeNull();
  });

  it("is consistent with parseWorkspaceURI for file ID extraction", () => {
    const uris = [
      "workspace://abc#image/png",
      "workspace://abc",
      "workspace://abc#",
      "workspace://550e8400-e29b-41d4-a716-446655440000#text/plain",
    ];

    for (const uri of uris) {
      const fullParse = parseWorkspaceURI(uri);
      const idOnly = parseWorkspaceFileID(uri);
      expect(idOnly).toBe(fullParse?.fileID ?? null);
    }
  });
});
