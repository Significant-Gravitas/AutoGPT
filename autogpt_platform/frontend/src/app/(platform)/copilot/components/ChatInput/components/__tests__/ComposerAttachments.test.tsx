import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Attachment } from "../../../../helpers/workspaceAttachments";
import { ComposerAttachments } from "../ComposerAttachments";

const photo: Attachment = {
  kind: "local",
  file: new File(["png"], "photo.png", { type: "image/png" }),
};

const notes: Attachment = {
  kind: "local",
  file: new File(["x".repeat(2048)], "notes.txt", { type: "text/plain" }),
};

const report: Attachment = {
  kind: "workspace",
  fileId: "file-1",
  name: "report.pdf",
  mimeType: "application/pdf",
};

const folder: Attachment = {
  kind: "folder",
  folderId: "fld-1",
  name: "Q3",
  fileCount: 2,
  subfolderCount: 1,
};

beforeEach(() => {
  URL.createObjectURL = vi.fn(() => "blob:photo");
  URL.revokeObjectURL = vi.fn();
});

afterEach(() => {
  cleanup();
});

describe("ComposerAttachments", () => {
  it("renders nothing when there are no attachments", () => {
    const { container } = render(
      <ComposerAttachments attachments={[]} onRemove={vi.fn()} />,
    );
    expect(container.textContent).toBe("");
  });

  it("shows a local image as a thumbnail and other files as cards", async () => {
    render(
      <ComposerAttachments
        attachments={[photo, notes, report, folder]}
        onRemove={vi.fn()}
      />,
    );

    const thumbnail = await screen.findByRole("img", { name: "photo.png" });
    expect(thumbnail.getAttribute("src")).toBe("blob:photo");
    expect(screen.getByText("notes.txt")).toBeDefined();
    expect(screen.getByText("Text · 2.0 KB")).toBeDefined();
    expect(screen.getByText("report.pdf")).toBeDefined();
    expect(screen.getByText("PDF")).toBeDefined();
    expect(screen.getByText("Folder · 2 files")).toBeDefined();
  });

  it("removes the attachment at its index", () => {
    const onRemove = vi.fn();
    render(
      <ComposerAttachments attachments={[photo, report]} onRemove={onRemove} />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Remove report.pdf" }));
    expect(onRemove).toHaveBeenCalledWith(1);
  });

  it("frees the preview when the image is removed", () => {
    const { rerender } = render(
      <ComposerAttachments attachments={[photo]} onRemove={vi.fn()} />,
    );
    rerender(<ComposerAttachments attachments={[]} onRemove={vi.fn()} />);
    // AnimatePresence keeps the exiting card mounted for its exit; the URL
    // is freed once it unmounts.
    cleanup();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:photo");
  });
});
