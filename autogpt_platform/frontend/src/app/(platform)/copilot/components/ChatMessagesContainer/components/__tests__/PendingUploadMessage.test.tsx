import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PENDING_UPLOAD_MESSAGE_ID,
  PendingUploadMessage,
} from "../PendingUploadMessage";

const flagState = vi.hoisted(() => ({ artifacts: false }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.ARTIFACTS ? flagState.artifacts : false,
  };
});

vi.mock("../ThinkingIndicator", () => ({
  ThinkingIndicator: ({ statusMessage }: { statusMessage?: string | null }) => (
    <div data-testid="thinking-indicator">{statusMessage}</div>
  ),
}));

const attachments = [
  {
    name: "talk.pdf",
    mediaType: "application/pdf",
    sizeBytes: 3 * 1024 * 1024,
    isUploading: true,
  },
  {
    name: "icon.png",
    mediaType: "image/png",
    sizeBytes: 2048,
    isUploading: true,
  },
  {
    name: "tiny.txt",
    mediaType: "text/plain",
    sizeBytes: 12,
    isUploading: true,
  },
  { name: "notes.md", mediaType: "text/markdown", isUploading: false },
];

afterEach(() => {
  cleanup();
  flagState.artifacts = false;
});

describe("PendingUploadMessage", () => {
  it("renders artifact-style cards with kind and size when artifacts are on", () => {
    flagState.artifacts = true;
    render(
      <PendingUploadMessage
        pendingSend={{ text: "look at these", attachments }}
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

  it("falls back to plain file cards showing the media type when artifacts are off", () => {
    render(<PendingUploadMessage pendingSend={{ text: "", attachments }} />);

    expect(screen.getByText("application/pdf")).toBeDefined();
    expect(screen.getByText("text/markdown")).toBeDefined();
  });

  it("carries the message id the tail spacer measures", () => {
    render(<PendingUploadMessage pendingSend={{ text: "hi", attachments }} />);

    expect(
      screen
        .getByTestId("pending-upload-message")
        .getAttribute("data-message-id"),
    ).toBe(PENDING_UPLOAD_MESSAGE_ID);
  });

  it("reads Sending when nothing is left to upload", () => {
    render(
      <PendingUploadMessage
        pendingSend={{
          text: "hi",
          attachments: attachments.filter((a) => !a.isUploading),
        }}
      />,
    );

    expect(screen.getByTestId("thinking-indicator").textContent).toBe(
      "Sending…",
    );
  });

  it("announces the upload status to screen readers", () => {
    render(<PendingUploadMessage pendingSend={{ text: "hi", attachments }} />);

    const status = screen.getByRole("status");
    expect(status.textContent).toBe("Uploading 3 files…");
    expect(status.getAttribute("aria-live")).toBe("polite");
    // The elapsed timer must stay outside the live region, or every tick
    // would re-announce the upload.
    expect(status.textContent).not.toContain("thinking");
  });
});
