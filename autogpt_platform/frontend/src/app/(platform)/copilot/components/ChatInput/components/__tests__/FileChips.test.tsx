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
