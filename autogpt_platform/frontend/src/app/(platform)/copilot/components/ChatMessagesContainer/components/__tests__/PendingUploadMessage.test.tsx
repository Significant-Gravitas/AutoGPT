import {
  cleanup,
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { PendingUploadAttachment } from "../../../../copilotStreamStore";
import {
  PENDING_UPLOAD_MESSAGE_ID,
  PendingUploadMessage,
} from "../PendingUploadMessage";

const { retryPendingUploadMock, removePendingUploadMock } = vi.hoisted(() => ({
  retryPendingUploadMock: vi.fn(),
  removePendingUploadMock: vi.fn(),
}));

vi.mock("../../../../helpers/sendWithAttachments", () => ({
  retryPendingUpload: retryPendingUploadMock,
  removePendingUpload: removePendingUploadMock,
}));

vi.mock("../ThinkingIndicator", () => ({
  ThinkingIndicator: ({ statusMessage }: { statusMessage?: string | null }) => (
    <div data-testid="thinking-indicator">{statusMessage}</div>
  ),
}));

const attachments: PendingUploadAttachment[] = [
  {
    id: "talk",
    name: "talk.pdf",
    mediaType: "application/pdf",
    sizeBytes: 3 * 1024 * 1024,
    status: "uploading",
  },
  {
    id: "icon",
    name: "icon.png",
    mediaType: "image/png",
    sizeBytes: 2048,
    status: "uploading",
  },
  {
    id: "tiny",
    name: "tiny.txt",
    mediaType: "text/plain",
    sizeBytes: 12,
    status: "uploading",
  },
  {
    id: "notes",
    name: "notes.md",
    mediaType: "text/markdown",
    status: "uploaded",
  },
];

function pending(list: PendingUploadAttachment[]) {
  return { id: "pending", text: "hi", attachments: list };
}

function cardFor(name: string) {
  const card = screen
    .getAllByTestId("pending-upload-attachment")
    .find((c) => within(c).queryByText(name));
  if (!card) throw new Error(`No card for ${name}`);
  return card;
}

afterEach(() => {
  cleanup();
});

describe("PendingUploadMessage", () => {
  it("renders artifact-style cards with kind and size", () => {
    render(
      <PendingUploadMessage
        pendingSend={{ id: "pending", text: "look at these", attachments }}
      />,
    );

    expect(screen.getByText("look at these")).toBeDefined();
    expect(screen.getByText("talk.pdf")).toBeDefined();
    expect(screen.getByText(/3\.0 MB/)).toBeDefined();
    expect(screen.getByText(/2\.0 KB/)).toBeDefined();
    expect(screen.getByText(/12 B/)).toBeDefined();
    expect(screen.getByText("notes.md")).toBeDefined();
    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Uploading 3 files…",
    );
  });

  it("carries the message id the tail spacer measures", () => {
    render(<PendingUploadMessage pendingSend={pending(attachments)} />);

    expect(
      screen
        .getByTestId("pending-upload-message")
        .getAttribute("data-message-id"),
    ).toBe(PENDING_UPLOAD_MESSAGE_ID);
  });

  it("reads Sending when nothing is left to upload", () => {
    render(
      <PendingUploadMessage
        pendingSend={pending(
          attachments.filter((a) => a.status === "uploaded"),
        )}
      />,
    );

    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Sending…",
    );
  });

  it("announces the upload status to screen readers", () => {
    render(<PendingUploadMessage pendingSend={pending(attachments)} />);

    const status = screen.getByRole("status");
    expect(status.textContent).toBe("Uploading 3 files…");
    expect(status.getAttribute("aria-live")).toBe("polite");
    // The elapsed timer must stay outside the live region, or every tick
    // would re-announce the upload.
    expect(status.textContent).not.toContain("thinking");
  });

  it("names each file's own upload status", () => {
    render(<PendingUploadMessage pendingSend={pending(attachments)} />);

    expect(within(cardFor("talk.pdf")).getByText("Uploading")).toBeDefined();
    expect(within(cardFor("notes.md")).getByText("Uploaded")).toBeDefined();
  });

  it("marks a failed file and offers Retry and Remove for it", () => {
    render(
      <PendingUploadMessage
        pendingSend={pending([
          { ...attachments[0], status: "failed" },
          attachments[3],
        ])}
      />,
    );

    const failed = cardFor("talk.pdf");
    expect(within(failed).getByText("Upload failed")).toBeDefined();
    expect(within(failed).queryByText(/3\.0 MB/)).toBeNull();
    fireEvent.click(
      within(failed).getByRole("button", { name: "Retry talk.pdf" }),
    );
    fireEvent.click(
      within(failed).getByRole("button", { name: "Remove talk.pdf" }),
    );
    expect(retryPendingUploadMock).toHaveBeenCalledWith("talk");
    expect(removePendingUploadMock).toHaveBeenCalledWith("talk");
    expect(
      within(cardFor("notes.md")).queryByRole("button", { name: /Retry/ }),
    ).toBeNull();
  });

  it("says what to do about a failed file instead of narrating progress", () => {
    render(
      <PendingUploadMessage
        pendingSend={pending([{ ...attachments[0], status: "failed" }])}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe(
      "Couldn't upload talk.pdf. Retry or remove it to send your message.",
    );
    expect(screen.queryByTestId("thinking-indicator")).toBeNull();
  });

  it("keeps narrating while another file still uploads", () => {
    render(
      <PendingUploadMessage
        pendingSend={pending([
          { ...attachments[0], status: "failed" },
          { ...attachments[1], status: "failed" },
          attachments[2],
        ])}
      />,
    );

    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Uploading 1 file…",
    );
  });

  it("counts several failed files", () => {
    render(
      <PendingUploadMessage
        pendingSend={pending([
          { ...attachments[0], status: "failed" },
          { ...attachments[1], status: "failed" },
        ])}
      />,
    );

    expect(screen.getByRole("status").textContent).toBe(
      "Couldn't upload 2 files. Retry or remove them to send your message.",
    );
  });
});
