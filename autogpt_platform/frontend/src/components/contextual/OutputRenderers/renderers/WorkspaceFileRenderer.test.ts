import { describe, expect, it } from "vitest";
import {
  parseWorkspaceURI,
  parseWorkspaceFileID,
  isWorkspaceURI,
  buildWorkspaceURI,
} from "@/lib/workspace-uri";
import { workspaceFileRenderer } from "./WorkspaceFileRenderer";

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

describe("isWorkspaceURI", () => {
  it("returns true for workspace URIs", () => {
    expect(isWorkspaceURI("workspace://abc")).toBe(true);
    expect(isWorkspaceURI("workspace://abc#image/png")).toBe(true);
  });

  it("returns false for non-workspace values", () => {
    expect(isWorkspaceURI("https://example.com")).toBe(false);
    expect(isWorkspaceURI("")).toBe(false);
    expect(isWorkspaceURI(null)).toBe(false);
    expect(isWorkspaceURI(undefined)).toBe(false);
    expect(isWorkspaceURI(123)).toBe(false);
  });
});

describe("buildWorkspaceURI", () => {
  it("builds URI with mime type", () => {
    expect(buildWorkspaceURI("file-123", "image/png")).toBe(
      "workspace://file-123#image/png",
    );
  });

  it("builds URI without mime type", () => {
    expect(buildWorkspaceURI("file-123")).toBe("workspace://file-123");
  });

  it("roundtrips with parseWorkspaceURI", () => {
    const uri = buildWorkspaceURI("file-abc", "text/plain");
    const parsed = parseWorkspaceURI(uri);
    expect(parsed).toEqual({ fileID: "file-abc", mimeType: "text/plain" });
  });
});

describe("workspaceFileRenderer.getDownloadContent", () => {
  it("returns auth-proxied URL without share token", () => {
    const result = workspaceFileRenderer.getDownloadContent(
      "workspace://file-123#image/png",
    );
    expect(result).not.toBeNull();
    expect(result!.data).toBe(
      "/api/proxy/api/workspace/files/file-123/download",
    );
  });

  it("returns public share URL when share token is in metadata", () => {
    const result = workspaceFileRenderer.getDownloadContent(
      "workspace://file-123#image/png",
      { shareToken: "abc-token-123" },
    );
    expect(result).not.toBeNull();
    expect(result!.data).toBe(
      "/api/proxy/api/public/shared/abc-token-123/files/file-123/download",
    );
  });
});
