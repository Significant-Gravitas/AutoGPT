import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { FileUIPart } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../../store";
import { UserMessageAttachments } from "../UserMessageAttachments";

const { downloadArtifactMock } = vi.hoisted(() => ({
  downloadArtifactMock: vi.fn(() => Promise.resolve()),
}));

vi.mock("../../../ArtifactPanel/downloadArtifact", () => ({
  downloadArtifact: downloadArtifactMock,
}));

const PHOTO_ID = "550e8400-e29b-41d4-a716-446655440000";
const REPORT_ID = "550e8400-e29b-41d4-a716-446655440001";
const SHEET_ID = "550e8400-e29b-41d4-a716-446655440002";

function filePart(id: string, filename: string, mediaType: string) {
  return {
    type: "file",
    url: `/api/proxy/api/workspace/files/${id}/download`,
    filename,
    mediaType,
    sizeBytes: 1536,
  } as FileUIPart;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  useCopilotUIStore.getState().closeArtifactPanel();
});

describe("UserMessageAttachments", () => {
  it("renders nothing without attachments", () => {
    const { container } = render(
      <UserMessageAttachments files={[]} folders={[]} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("shows an image as a thumbnail and a document as a card with type and size", () => {
    render(
      <UserMessageAttachments
        files={[
          filePart(PHOTO_ID, "photo.png", "image/png"),
          filePart(REPORT_ID, "report.pdf", "application/pdf"),
        ]}
        folders={[]}
      />,
    );

    const thumbnail = screen.getByRole("img", { name: "photo.png" });
    expect(thumbnail.getAttribute("src")).toBe(
      `/api/proxy/api/workspace/files/${PHOTO_ID}/download`,
    );
    expect(screen.getByText("report.pdf")).toBeDefined();
    expect(screen.getByText("PDF · 1.5 KB")).toBeDefined();
  });

  it("opens a file the panel can show", () => {
    render(
      <UserMessageAttachments
        files={[filePart(REPORT_ID, "report.pdf", "application/pdf")]}
        folders={[]}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /report\.pdf/ }));

    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact,
    ).toMatchObject({ id: REPORT_ID, title: "report.pdf" });
    expect(downloadArtifactMock).not.toHaveBeenCalled();
  });

  it("downloads a file the panel cannot show", () => {
    render(
      <UserMessageAttachments
        files={[
          filePart(
            SHEET_ID,
            "budget.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          ),
        ]}
        folders={[]}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /budget\.xlsx/ }));

    expect(downloadArtifactMock).toHaveBeenCalledWith(
      expect.objectContaining({ id: SHEET_ID }),
    );
  });

  it("links a folder to Files, but not in the share viewer", () => {
    const folder = { id: "fld-1", name: "Reports", fileCount: 3 };
    const { rerender } = render(
      <UserMessageAttachments files={[]} folders={[folder]} />,
    );

    expect(screen.getByText("Folder · 3 files")).toBeDefined();
    expect(
      screen.getByRole("link", { name: /Reports/ }).getAttribute("href"),
    ).toBe("/artifacts?folder=fld-1");

    rerender(<UserMessageAttachments files={[]} folders={[folder]} readOnly />);
    expect(screen.queryByRole("link")).toBeNull();
    expect(screen.getByText("Reports")).toBeDefined();
  });

  it("does not call a folder with no direct files empty", () => {
    render(
      <UserMessageAttachments
        files={[]}
        folders={[{ id: "fld-1", name: "Reports", fileCount: 0 }]}
      />,
    );
    expect(screen.getByText("Folder")).toBeDefined();
    expect(screen.queryByText(/Empty/)).toBeNull();
  });
});
