import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ArtifactRef } from "../../../../store";
import { AttachmentPreview } from "../AttachmentPreview";

const registerSpy = vi.fn();
const openSpy = vi.fn();

vi.mock("../../../../store", () => {
  function useCopilotUIStore<T>(selector: (state: unknown) => T): T {
    const state = {
      artifactPanel: { isOpen: false, activeArtifact: null },
      openArtifact: openSpy,
      registerArtifactForAutoOpen: registerSpy,
    };
    return selector(state);
  }
  return { useCopilotUIStore };
});

const downloadArtifactMock = vi.fn(() => Promise.resolve());
vi.mock("../../../ArtifactPanel/downloadArtifact", () => ({
  downloadArtifact: (...args: unknown[]) =>
    downloadArtifactMock(...(args as [])),
}));

const IMAGE: ArtifactRef = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  title: "pane-icon-2.png",
  mimeType: "image/png",
  origin: "user-upload",
  sourceUrl:
    "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440000/download",
};

const PDF: ArtifactRef = {
  ...IMAGE,
  id: "660e8400-e29b-41d4-a716-446655440000",
  title: "AsyncIO_ Python Users Berlin.pdf",
  mimeType: "application/pdf",
  sourceUrl:
    "/api/proxy/api/workspace/files/660e8400-e29b-41d4-a716-446655440000/download",
};

const ZIP: ArtifactRef = {
  ...IMAGE,
  id: "770e8400-e29b-41d4-a716-446655440000",
  title: "bundle.zip",
  mimeType: "application/zip",
};

beforeEach(() => {
  registerSpy.mockClear();
  openSpy.mockClear();
  downloadArtifactMock.mockClear();
});

describe("AttachmentPreview", () => {
  it("renders an image attachment as a thumbnail that opens the panel", () => {
    render(<AttachmentPreview artifact={IMAGE} />);
    const img = screen.getByRole("img", { name: "pane-icon-2.png" });
    expect(img.getAttribute("src")).toBe(IMAGE.sourceUrl);
    fireEvent.click(screen.getByTestId("attachment-preview-image"));
    expect(openSpy).toHaveBeenCalledWith(IMAGE);
    expect(registerSpy).toHaveBeenCalledWith(IMAGE);
  });

  it("falls back to the file card when the image fails to load", () => {
    render(<AttachmentPreview artifact={IMAGE} />);
    fireEvent.error(screen.getByRole("img", { name: "pane-icon-2.png" }));
    expect(screen.queryByTestId("attachment-preview-image")).toBeNull();
    expect(screen.getByTestId("attachment-preview-file")).toBeTruthy();
    expect(screen.getByText("pane-icon-2.png")).toBeTruthy();
  });

  it("renders a non-image attachment as a compact card with its type label", () => {
    render(<AttachmentPreview artifact={PDF} />);
    expect(screen.getByText("AsyncIO_ Python Users Berlin.pdf")).toBeTruthy();
    expect(screen.getByText("PDF")).toBeTruthy();
    fireEvent.click(screen.getByTestId("attachment-preview-file"));
    expect(openSpy).toHaveBeenCalledWith(PDF);
  });

  it("downloads instead of opening for a non-openable attachment", () => {
    render(<AttachmentPreview artifact={ZIP} />);
    fireEvent.click(screen.getByTestId("attachment-preview-file"));
    expect(openSpy).not.toHaveBeenCalled();
    expect(downloadArtifactMock).toHaveBeenCalledWith(ZIP);
  });

  it("does not register for auto-open in readOnly mode", () => {
    render(<AttachmentPreview artifact={IMAGE} readOnly />);
    expect(registerSpy).not.toHaveBeenCalled();
  });
});
