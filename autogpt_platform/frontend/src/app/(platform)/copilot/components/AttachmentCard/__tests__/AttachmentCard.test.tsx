import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AttachmentCard } from "../AttachmentCard";
import { attachmentDetail, formatSize } from "../helpers";

afterEach(() => {
  cleanup();
});

describe("AttachmentCard", () => {
  it("shows an image as a thumbnail", () => {
    render(
      <AttachmentCard
        size="message"
        name="photo.png"
        mediaType="image/png"
        previewUrl="blob:photo"
      />,
    );

    const image = screen.getByRole("img", { name: "photo.png" });
    expect(image.getAttribute("src")).toBe("blob:photo");
  });

  it("falls back to the file card when the image cannot be drawn", () => {
    render(
      <AttachmentCard
        size="message"
        name="photo.png"
        mediaType="image/png"
        sizeBytes={2048}
        previewUrl="blob:broken"
      />,
    );

    fireEvent.error(screen.getByRole("img", { name: "photo.png" }));

    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("Image · 2.0 KB")).toBeDefined();
  });

  it("shows any other file as a card with its name, type and size", () => {
    render(
      <AttachmentCard
        size="message"
        name="report.pdf"
        mediaType="application/pdf"
        sizeBytes={1.2 * 1024 * 1024}
      />,
    );

    expect(screen.getByText("report.pdf")).toBeDefined();
    expect(screen.getByText("PDF · 1.2 MB")).toBeDefined();
  });

  it("names the remove button after the file", () => {
    const onRemove = vi.fn();
    render(
      <AttachmentCard
        size="composer"
        name="photo.png"
        mediaType="image/png"
        previewUrl="blob:photo"
        onRemove={onRemove}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Remove photo.png" }));
    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("spins while uploading and names the state", () => {
    render(
      <AttachmentCard
        size="message"
        name="photo.png"
        mediaType="image/png"
        status="uploading"
      />,
    );

    expect(screen.getByTestId("attachment-card").dataset.status).toBe(
      "uploading",
    );
    expect(screen.getByText("Uploading")).toBeDefined();
  });

  it("turns a failed image into a card with the reason and a Retry", () => {
    const onRetry = vi.fn();
    render(
      <AttachmentCard
        size="message"
        name="photo.png"
        mediaType="image/png"
        previewUrl="blob:photo"
        status="failed"
        onRetry={onRetry}
        onRemove={vi.fn()}
      />,
    );

    expect(screen.queryByRole("img")).toBeNull();
    expect(screen.getByText("Upload failed")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Retry photo.png" }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(
      screen.getByRole("button", { name: "Remove photo.png" }),
    ).toBeDefined();
  });

  it("opens the file from the card body", () => {
    const onOpen = vi.fn();
    render(
      <AttachmentCard
        size="message"
        name="report.pdf"
        mediaType="application/pdf"
        onOpen={onOpen}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /report\.pdf/ }));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });
});

describe("attachment details", () => {
  it("leaves the size out until it is known", () => {
    expect(attachmentDetail("application/pdf", "report.pdf")).toBe("PDF");
  });

  it("formats sizes the way the artifact cards do", () => {
    expect(formatSize(12)).toBe("12 B");
    expect(formatSize(2048)).toBe("2.0 KB");
    expect(formatSize(3 * 1024 * 1024)).toBe("3.0 MB");
    expect(formatSize(undefined)).toBe("");
  });
});

describe("attachment type labels", () => {
  it("names office documents the panel can only download", () => {
    expect(
      attachmentDetail(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "budget.xlsx",
      ),
    ).toBe("Spreadsheet");
  });

  it("calls an unknown type a file", () => {
    expect(attachmentDetail("application/zip", "bundle.zip", 2048)).toBe(
      "File · 2.0 KB",
    );
  });
});
