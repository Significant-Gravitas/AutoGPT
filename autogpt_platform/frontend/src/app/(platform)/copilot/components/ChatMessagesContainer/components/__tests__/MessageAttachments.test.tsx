import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { FileUIPart } from "ai";
import { useCopilotUIStore } from "../../../../store";
import { MessageAttachments } from "../MessageAttachments";

vi.mock("../../../../components/ArtifactCard/ArtifactCard", () => ({
  ArtifactCard: () => <div data-testid="artifact-card" />,
}));

const FILE_ID = "550e8400-e29b-41d4-a716-446655440000";

const FILE_URL = `/api/proxy/api/workspace/files/${FILE_ID}/download`;

function filePart(
  url: string,
  filename = "report.pdf",
  mediaType = "application/pdf",
): FileUIPart {
  return { type: "file", url, filename, mediaType } as FileUIPart;
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

  it("shows an image the user sent as a picture that opens in the panel", () => {
    render(
      <MessageAttachments
        files={[filePart(FILE_URL, "photo.jpg", "image/jpeg")]}
        isUser
      />,
    );

    expect(screen.queryByTestId("artifact-card")).toBeNull();
    const image = screen.getByRole("img", { name: "photo.jpg" });
    expect(image.getAttribute("src")).toBe(FILE_URL);

    fireEvent.click(screen.getByRole("button", { name: "Open photo.jpg" }));
    expect(useCopilotUIStore.getState().artifactPanel.activeArtifact?.id).toBe(
      FILE_ID,
    );
  });

  it("falls back to the card when the picture fails to load", () => {
    render(
      <MessageAttachments
        files={[filePart(FILE_URL, "photo.jpg", "image/jpeg")]}
        isUser
      />,
    );

    fireEvent.error(screen.getByRole("img", { name: "photo.jpg" }));
    expect(screen.getByTestId("artifact-card")).toBeDefined();
  });

  it("keeps the card for an image Otto sent", () => {
    render(
      <MessageAttachments
        files={[filePart(FILE_URL, "chart.png", "image/png")]}
      />,
    );

    expect(screen.getByTestId("artifact-card")).toBeDefined();
    expect(screen.queryByRole("img")).toBeNull();
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
