import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

describe("FileChips — image thumbnails", () => {
  const localImage: Attachment = {
    kind: "local",
    file: new File(["png"], "photo.png", { type: "image/png" }),
  };
  const workspaceImage: Attachment = {
    kind: "workspace",
    fileId: "550e8400-e29b-41d4-a716-446655440000",
    name: "diagram.jpg",
    mimeType: "image/jpeg",
  };

  it("renders a thumbnail instead of a text chip for a local image", () => {
    const createObjectURL = vi.fn(() => "blob:local-photo");
    const revokeObjectURL = vi.fn();
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;

    try {
      const { unmount } = render(
        <FileChips attachments={[localImage]} onRemove={vi.fn()} />,
      );
      const img = screen.getByTestId("attachment-thumbnail");
      expect(img.getAttribute("src")).toBe("blob:local-photo");
      expect(img.getAttribute("alt")).toBe("photo.png");
      expect(screen.queryByText("photo.png")).toBeNull();
      expect(
        screen.getByRole("button", { name: /remove photo\.png/i }),
      ).toBeTruthy();

      unmount();
      expect(revokeObjectURL).toHaveBeenCalledWith("blob:local-photo");
    } finally {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    }
  });

  it("renders a workspace image through the preview endpoint", () => {
    render(<FileChips attachments={[workspaceImage]} onRemove={vi.fn()} />);
    const img = screen.getByTestId("attachment-thumbnail");
    expect(img.getAttribute("src")).toContain(
      "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/preview",
    );
  });

  it("keeps the text chip for non-image files", () => {
    render(
      <FileChips attachments={[workspaceAttachment]} onRemove={vi.fn()} />,
    );
    expect(screen.queryByTestId("attachment-thumbnail")).toBeNull();
    expect(screen.getByText("report.pdf")).toBeTruthy();
  });

  describe("thumbnail load failure", () => {
    const originalCreate = URL.createObjectURL;
    const originalRevoke = URL.revokeObjectURL;

    beforeEach(() => {
      URL.createObjectURL = vi.fn(() => "blob:local-photo");
      URL.revokeObjectURL = vi.fn();
    });

    afterEach(() => {
      URL.createObjectURL = originalCreate;
      URL.revokeObjectURL = originalRevoke;
    });

    it("falls back to the filename chip for a local image and keeps its removal index", () => {
      const onRemove = vi.fn();
      render(
        <FileChips
          attachments={[workspaceAttachment, localImage]}
          onRemove={onRemove}
        />,
      );
      fireEvent.error(screen.getByTestId("attachment-thumbnail"));

      expect(screen.queryByTestId("attachment-thumbnail")).toBeNull();
      expect(screen.getByText("photo.png")).toBeTruthy();
      fireEvent.click(
        screen.getByRole("button", { name: /remove photo\.png/i }),
      );
      expect(onRemove).toHaveBeenCalledWith(1);
    });

    it("falls back to the filename chip for a workspace image", () => {
      const onRemove = vi.fn();
      render(<FileChips attachments={[workspaceImage]} onRemove={onRemove} />);
      fireEvent.error(screen.getByTestId("attachment-thumbnail"));

      expect(screen.queryByTestId("attachment-thumbnail")).toBeNull();
      expect(screen.getByText("diagram.jpg")).toBeTruthy();
      fireEvent.click(
        screen.getByRole("button", { name: /remove diagram\.jpg/i }),
      );
      expect(onRemove).toHaveBeenCalledWith(0);
    });

    it("shows the upload spinner on the fallback chip while uploading", () => {
      const { rerender } = render(
        <FileChips attachments={[localImage]} onRemove={vi.fn()} />,
      );
      fireEvent.error(screen.getByTestId("attachment-thumbnail"));
      rerender(
        <FileChips
          attachments={[localImage]}
          onRemove={vi.fn()}
          isUploading={true}
        />,
      );

      expect(screen.getByText("photo.png")).toBeTruthy();
      expect(
        screen.queryByRole("button", { name: /remove photo\.png/i }),
      ).toBeNull();
    });

    it("does not carry a failure over to the attachment that replaces it", () => {
      const replacement: Attachment = {
        kind: "local",
        file: new File(["gif"], "other.gif", { type: "image/gif" }),
      };
      const { rerender } = render(
        <FileChips attachments={[localImage]} onRemove={vi.fn()} />,
      );
      fireEvent.error(screen.getByTestId("attachment-thumbnail"));
      expect(screen.queryByTestId("attachment-thumbnail")).toBeNull();

      rerender(<FileChips attachments={[replacement]} onRemove={vi.fn()} />);

      expect(
        screen.getByTestId("attachment-thumbnail").getAttribute("alt"),
      ).toBe("other.gif");
      expect(screen.queryByText("other.gif")).toBeNull();
    });
  });
});
