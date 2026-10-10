import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { resetNavigation } from "./navigation-mock";
import { http, HttpResponse } from "msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFoldersMockHandler,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ARTIFACTS_PAGE: "artifacts-page",
    AUTOGPT_NEW_LAYOUT: "autogpt-new-layout",
    HIRE_EXPERTS: "hire-experts",
  },
  useGetFlag: (flag: string) => flag !== "autogpt-new-layout",
  useFlagStatus: () => ({ enabled: true, ready: true }),
}));

vi.mock("next/navigation", async () => {
  const { navigationMock } = await import("./navigation-mock");
  return navigationMock();
});

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

import ArtifactsPage from "../page";

const PPTX_MIME =
  "application/vnd.openxmlformats-officedocument.presentationml.presentation";

let downloadRequests: string[] = [];

beforeEach(() => {
  resetNavigation();
  downloadRequests = [];
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getListWorkspaceFoldersMockHandler({ folders: [] }),
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: 0,
      limit_bytes: 1_000_000_000,
      used_percent: 0,
      file_count: 0,
    }),
    http.get("/api/proxy/api/workspace/files/:id/download", ({ params }) => {
      downloadRequests.push(String(params.id));
      return HttpResponse.text("PK\u0003\u0004 binary zip bytes");
    }),
  );
});

function makeFile(
  overrides: Partial<WorkspaceFileItem> = {},
): WorkspaceFileItem {
  return {
    id: "file-base",
    name: "base.txt",
    path: "/base.txt",
    mime_type: "text/plain",
    size_bytes: 1024,
    metadata: {},
    origin: "generated",
    created_at: "2026-05-01T00:00:00Z",
    ...overrides,
  };
}

function useFilesHandler(files: WorkspaceFileItem[]) {
  server.use(
    getListWorkspaceFilesMockHandler({
      files,
      offset: 0,
      has_more: false,
    }),
  );
}

async function openViewer() {
  fireEvent.click(await screen.findByTestId("artifacts-card-open"));
  return screen.findByTestId("file-viewer");
}

function expectDownloadLink(viewer: HTMLElement, fileId: string) {
  const link = within(viewer).getByRole("link", { name: /download/i });
  expect(link.getAttribute("href")).toBe(
    `/api/proxy/api/workspace/files/${fileId}/download`,
  );
}

describe("ArtifactsPage - office files in the viewer", () => {
  test.each([
    ["a text MIME", "text/plain"],
    ["the openxml MIME", PPTX_MIME],
  ])(
    "a .pptx with %s shows the slide thumbnail, never the raw bytes",
    async (_label, mime) => {
      useFilesHandler([
        makeFile({ id: "deck1", name: "deck.pptx", mime_type: mime }),
      ]);

      render(<ArtifactsPage />);
      const viewer = await openViewer();

      const img = within(viewer).getByTestId("file-viewer-office-preview");
      expect(img.getAttribute("src")).toContain(
        "/api/proxy/api/workspace/files/deck1/preview?w=800",
      );
      expect(img.getAttribute("alt")).toBe("deck.pptx");
      expectDownloadLink(viewer, "deck1");
      expect(viewer.querySelector("pre")).toBeNull();
      expect(within(viewer).queryByText(/can't be previewed/i)).toBeNull();
      expect(downloadRequests).toEqual([]);
    },
  );

  test("a deck without a thumbnail falls back to the download prompt", async () => {
    useFilesHandler([
      makeFile({ id: "deck2", name: "deck.pptx", mime_type: PPTX_MIME }),
    ]);

    render(<ArtifactsPage />);
    const viewer = await openViewer();

    fireEvent.error(within(viewer).getByTestId("file-viewer-office-preview"));

    expect(
      await within(viewer).findByText(/can't be previewed/i),
    ).toBeDefined();
    expect(
      within(viewer).queryByTestId("file-viewer-office-preview"),
    ).toBeNull();
    expectDownloadLink(viewer, "deck2");
  });

  test("a .docx with a text MIME shows the thumbnail and falls back the same way", async () => {
    useFilesHandler([
      makeFile({ id: "doc1", name: "report.docx", mime_type: "text/plain" }),
    ]);

    render(<ArtifactsPage />);
    const viewer = await openViewer();

    const img = within(viewer).getByTestId("file-viewer-office-preview");
    expect(img.getAttribute("src")).toContain(
      "/api/proxy/api/workspace/files/doc1/preview?w=800",
    );
    expect(viewer.querySelector("pre")).toBeNull();

    fireEvent.error(img);

    expect(
      await within(viewer).findByText(/can't be previewed/i),
    ).toBeDefined();
    expectDownloadLink(viewer, "doc1");
    expect(downloadRequests).toEqual([]);
  });

  test("a .pptx over the 50MB preview cap goes straight to the download prompt", async () => {
    useFilesHandler([
      makeFile({
        id: "big1",
        name: "huge.pptx",
        mime_type: "text/plain",
        size_bytes: 60_000_000,
      }),
    ]);

    render(<ArtifactsPage />);
    const viewer = await openViewer();

    expect(within(viewer).getByText(/can't be previewed/i)).toBeDefined();
    expect(
      within(viewer).queryByTestId("file-viewer-office-preview"),
    ).toBeNull();
    expect(viewer.querySelector("pre")).toBeNull();
    expectDownloadLink(viewer, "big1");
  });

  test("a legacy .ppt with a text MIME is download-only, not text", async () => {
    useFilesHandler([
      makeFile({ id: "old1", name: "old.ppt", mime_type: "text/plain" }),
    ]);

    render(<ArtifactsPage />);
    const viewer = await openViewer();

    expect(within(viewer).getByText(/can't be previewed/i)).toBeDefined();
    expect(viewer.querySelector("pre")).toBeNull();
    expect(downloadRequests).toEqual([]);
  });
});
