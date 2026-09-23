import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import type { FileUIPart } from "ai";
import { MessageAttachments } from "../MessageAttachments";

vi.mock("../../../../components/ArtifactCard/ArtifactCard", () => ({
  ArtifactCard: () => <div data-testid="artifact-card" />,
}));

const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

function filePart(url: string, filename = "report.pdf"): FileUIPart {
  return {
    type: "file",
    url,
    filename,
    mediaType: "application/pdf",
  } as FileUIPart;
}

describe("MessageAttachments", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders nothing for an empty file list", () => {
    const { container } = render(<MessageAttachments files={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders ArtifactCards without a feature flag", () => {
    render(
      <MessageAttachments
        files={[filePart(`/api/proxy/api/workspace/files/${FILE_ID}/download`)]}
      />,
    );
    expect(screen.getByTestId("artifact-card")).toBeDefined();
  });

  it("renders an attached folder as a chip, not an artifact card", () => {
    render(
      <MessageAttachments
        files={[]}
        folders={[{ id: "fld-1", name: "Reports", fileCount: 3 }]}
      />,
    );
    expect(screen.getByText("Reports")).toBeDefined();
    expect(screen.getByText("Folder · 3 files")).toBeDefined();
    expect(screen.queryByTestId("artifact-card")).toBeNull();
  });

  it("renders a folder-only message instead of nothing", () => {
    const { container } = render(
      <MessageAttachments
        files={[]}
        folders={[{ id: "fld-1", name: "Reports", fileCount: 0 }]}
      />,
    );
    expect(container.firstChild).not.toBeNull();
    // Zero direct files may still hold subfolders the part does not count.
    expect(screen.queryByText(/Empty/)).toBeNull();
  });

  it("falls back to the file card for non-workspace files", () => {
    render(
      <MessageAttachments
        files={[filePart("https://example.com/file.txt", "file.txt")]}
        isUser
      />,
    );
    expect(screen.getByText("file.txt")).toBeDefined();
  });
});
