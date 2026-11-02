import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Attachment } from "../../../../helpers/workspaceAttachments";
import { FileChips } from "../FileChips";

const localAttachment: Attachment = {
  kind: "local",
  file: new File(["hi"], "local.txt", { type: "text/plain" }),
};

const workspaceAttachment: Attachment = {
  kind: "workspace",
  fileId: "file-1",
  name: "report.pdf",
  mimeType: "application/pdf",
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("FileChips", () => {
  it("renders nothing when there are no attachments", () => {
    const { container } = render(
      <FileChips attachments={[]} onRemove={vi.fn()} />,
    );
    expect(container.textContent).toBe("");
  });

  it("renders a chip for each attachment", () => {
    render(
      <FileChips
        attachments={[localAttachment, workspaceAttachment]}
        onRemove={vi.fn()}
      />,
    );
    expect(screen.getByText("local.txt")).toBeTruthy();
    expect(screen.getByText("report.pdf")).toBeTruthy();
  });

  it("removes the attachment at its index when the remove button is clicked", () => {
    const onRemove = vi.fn();
    render(
      <FileChips
        attachments={[localAttachment, workspaceAttachment]}
        onRemove={onRemove}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /remove report\.pdf/i }),
    );
    expect(onRemove).toHaveBeenCalledWith(1);
  });

  it("shows a thumbnail of an attached image in place of its icon", () => {
    const photo: Attachment = {
      kind: "local",
      file: new File(["jpg"], "photo.jpg", { type: "image/jpeg" }),
    };
    const stored: Attachment = {
      kind: "workspace",
      fileId: "file-2",
      name: "shot.png",
      mimeType: "image/png",
    };
    const { container } = render(
      <FileChips
        attachments={[photo, stored, localAttachment]}
        onRemove={vi.fn()}
      />,
    );

    const sources = Array.from(container.querySelectorAll("img")).map((img) =>
      img.getAttribute("src"),
    );
    expect(sources).toHaveLength(2);
    expect(sources[0]).toMatch(/^blob:/);
    expect(sources[1]).toBe("/api/proxy/api/workspace/files/file-2/download");
    expect(screen.getByText("photo.jpg")).toBeTruthy();
  });

  it("falls back to the file icon when an image thumbnail fails to load", () => {
    const stored: Attachment = {
      kind: "workspace",
      fileId: "file-2",
      name: "shot.png",
      mimeType: "image/png",
    };
    const { container } = render(
      <FileChips attachments={[stored]} onRemove={vi.fn()} />,
    );

    const image = container.querySelector("img");
    expect(image).not.toBeNull();
    fireEvent.error(image!);

    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
    expect(screen.getByText("shot.png")).toBeTruthy();
  });

  it("shows a spinner instead of a remove button for local files while uploading", () => {
    render(
      <FileChips
        attachments={[localAttachment, workspaceAttachment]}
        onRemove={vi.fn()}
        isUploading={true}
      />,
    );
    // Local file: spinner replaces the remove button.
    expect(
      screen.queryByRole("button", { name: /remove local\.txt/i }),
    ).toBeNull();
    // Workspace file needs no upload, so it keeps its remove button.
    expect(
      screen.getByRole("button", { name: /remove report\.pdf/i }),
    ).toBeTruthy();
  });
});
