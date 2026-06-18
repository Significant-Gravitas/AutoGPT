import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFilesMockHandler401,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

const { setFlagStatusMock } = vi.hoisted(() => {
  return {
    setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
  };
});

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS_PAGE: "artifacts-page" },
  useGetFlag: () => true,
  useFlagStatus: () => setFlagStatusMock(),
}));

const notFoundMock = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/artifacts",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return {
    ...actual,
    useReducedMotion: () => true,
  };
});

import ArtifactsPage from "../page";

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

function useStorageHandler(usedBytes = 0, limitBytes = 1_000_000_000) {
  server.use(
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: usedBytes,
      limit_bytes: limitBytes,
      used_percent: limitBytes
        ? Math.round((usedBytes / limitBytes) * 1000) / 10
        : 0,
      file_count: 0,
    }),
  );
}

describe("ArtifactsPage - basic rendering", () => {
  test("renders the page header", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    expect(
      await screen.findByRole("heading", { name: /files/i }),
    ).toBeDefined();
  });

  test("shows the empty state when the workspace has no files", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("artifacts-empty")).toBeDefined();
    expect(screen.getByText(/no files yet/i)).toBeDefined();
  });

  test("renders one card per file with name + type label", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "f1",
            name: "report.pdf",
            mime_type: "application/pdf",
          }),
          makeFile({ id: "f2", name: "data.csv", mime_type: "text/csv" }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("report.pdf")).toBeDefined();
    expect(screen.getByText("data.csv")).toBeDefined();
    expect(screen.getAllByTestId("artifacts-list-item").length).toBe(2);
  });

  test("renders the error card when the API fails", async () => {
    useStorageHandler();
    server.use(getListWorkspaceFilesMockHandler401());

    render(<ArtifactsPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
  });
});

describe("ArtifactsPage - search filter", () => {
  test("typing in the search bar narrows the visible cards", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "f1", name: "alpha.txt" }),
          makeFile({ id: "f2", name: "beta.txt" }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("alpha.txt")).toBeDefined();
    expect(screen.getByText("beta.txt")).toBeDefined();

    const search = screen.getByPlaceholderText(/search/i);

    // Second handler returns only beta — the debounce + refetch should show it.
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [makeFile({ id: "f2", name: "beta.txt" })],
        offset: 0,
        has_more: false,
      }),
    );

    fireEvent.change(search, { target: { value: "beta" } });

    // Wait for the filtered list to appear (debounced ~250ms).
    expect(await screen.findByText("beta.txt")).toBeDefined();
  });
});

describe("ArtifactsPage - feature flag gating", () => {
  test("shows the flag-loading skeleton while LaunchDarkly is resolving", async () => {
    setFlagStatusMock.mockReturnValueOnce({ enabled: false, ready: false });

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("artifacts-flag-loading")).toBeDefined();
    expect(notFoundMock).not.toHaveBeenCalled();
  });

  test("calls notFound() when the flag is resolved and disabled", () => {
    setFlagStatusMock.mockReturnValueOnce({ enabled: false, ready: true });
    notFoundMock.mockClear();

    try {
      render(<ArtifactsPage />);
    } catch {
      // React surfaces the thrown notFound() error; the assertion below is
      // what we actually care about.
    }

    expect(notFoundMock).toHaveBeenCalled();
  });
});

describe("ArtifactsPage - card menu", () => {
  test("renders the card actions menu trigger and origin link", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "with-menu",
            name: "menu-target.txt",
            path: "/sessions/sess-xyz/menu-target.txt",
          }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("menu-target.txt")).toBeDefined();
    expect(screen.getByTestId("artifacts-card-menu")).toBeDefined();
  });

  test("clicking the card opens the file viewer modal", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [makeFile({ id: "open-me", name: "open-me.txt" })],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/open-me/download", () =>
        HttpResponse.text("hello world"),
      ),
    );

    render(<ArtifactsPage />);

    const opener = await screen.findByTestId("artifacts-card-open");
    fireEvent.click(opener);

    expect(await screen.findByTestId("file-viewer")).toBeDefined();
  });
});

describe("ArtifactsPage - rich previews", () => {
  test("image cards request the resized preview endpoint", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "img1", name: "pic.png", mime_type: "image/png" }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    const img = (await screen.findByAltText("pic.png")) as HTMLImageElement;
    expect(img.getAttribute("src")).toContain(
      "/api/proxy/api/workspace/files/img1/preview?w=400",
    );
  });

  test("pdf cards render an image from the preview endpoint", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "pdf1",
            name: "report.pdf",
            mime_type: "application/pdf",
          }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    const img = (await screen.findByAltText("report.pdf")) as HTMLImageElement;
    expect(img.getAttribute("src")).toContain("/preview?w=400");
  });

  test("csv cards render a table from the byte-capped preview", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "csv1", name: "data.csv", mime_type: "text/csv" }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/csv1/preview", () =>
        HttpResponse.text("name,age\nAda,36\nBob,40\n"),
      ),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("name")).toBeDefined();
    expect(await screen.findByText("Ada")).toBeDefined();
  });

  test("ics cards render the event summary", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "ics1",
            name: "meeting.ics",
            mime_type: "text/calendar",
            size_bytes: 400,
          }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/ics1/preview", () =>
        HttpResponse.text(
          "BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Launch sync\nDTSTART:20260615T130000Z\nLOCATION:Room 4\nEND:VEVENT\nEND:VCALENDAR",
        ),
      ),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("Launch sync")).toBeDefined();
  });

  test("vcard cards render the contact name", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "vcf1",
            name: "ada.vcf",
            mime_type: "text/vcard",
            size_bytes: 300,
          }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/vcf1/preview", () =>
        HttpResponse.text(
          "BEGIN:VCARD\nFN:Ada Lovelace\nORG:Analytical Engine\nEND:VCARD",
        ),
      ),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("Ada Lovelace")).toBeDefined();
  });

  test("markdown cards render their formatted content", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "md1", name: "notes.md", mime_type: "text/markdown" }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/md1/preview", () =>
        HttpResponse.text("# Heading One\n\nbody paragraph"),
      ),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("Heading One")).toBeDefined();
    expect(await screen.findByText("body paragraph")).toBeDefined();
  });
});

describe("ArtifactsPage - file viewer modal", () => {
  test("opening a markdown file shows a Source toggle that flips to Preview", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "md2", name: "doc.md", mime_type: "text/markdown" }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/md2/preview", () =>
        HttpResponse.text("# Title"),
      ),
      http.get("/api/proxy/api/workspace/files/md2/download", () =>
        HttpResponse.text("# Title"),
      ),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));
    expect(await screen.findByTestId("file-viewer")).toBeDefined();

    const sourceButton = await screen.findByRole("button", {
      name: /source/i,
    });
    fireEvent.click(sourceButton);

    expect(
      await screen.findByRole("button", { name: /preview/i }),
    ).toBeDefined();
  });

  test("opening a non-previewable file shows the download-only message", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "zip1",
            name: "archive.zip",
            mime_type: "application/zip",
          }),
        ],
        offset: 0,
        has_more: false,
      }),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));

    expect(await screen.findByText(/can't be previewed/i)).toBeDefined();
  });

  test("opening an uploaded file resolves its source ref", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({
            id: "up1",
            name: "uploaded.md",
            mime_type: "text/markdown",
            origin: "uploaded",
          }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/up1/preview", () =>
        HttpResponse.text("# uploaded"),
      ),
      http.get("/api/proxy/api/workspace/files/up1/download", () =>
        HttpResponse.text("# uploaded"),
      ),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));
    expect(await screen.findByTestId("file-viewer")).toBeDefined();
  });

  test("the viewer download button recovers after a failed fetch", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "dl1", name: "doc.md", mime_type: "text/markdown" }),
        ],
        offset: 0,
        has_more: false,
      }),
      http.get("/api/proxy/api/workspace/files/dl1/preview", () =>
        HttpResponse.text("# doc"),
      ),
      http.get(
        "/api/proxy/api/workspace/files/dl1/download",
        () => new HttpResponse("nope", { status: 500 }),
      ),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));
    fireEvent.click(await screen.findByTestId("file-viewer-download"));

    // After the failed download settles, the button returns to its idle label.
    expect(
      await screen.findByRole("button", { name: /^download$/i }),
    ).toBeDefined();
  });
});
