import { useOrgTeamStore } from "@/services/org-team/store";
import { afterEach, describe, expect, test, vi } from "vitest";

import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFilesMockHandler401,
  getListWorkspaceFoldersMockHandler,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

const { setFlagStatusMock, uploadFileDirectMock } = vi.hoisted(() => {
  return {
    setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
    uploadFileDirectMock: vi.fn(),
  };
});

// usePlatformChrome re-renders once on mount (isMounted guard), so per-test
// flag overrides must persist across renders; restore the default afterward.
afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  uploadFileDirectMock.mockReset();
});

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ARTIFACTS_PAGE: "artifacts-page",
    AUTOGPT_NEW_LAYOUT: "autogpt-new-layout",
  },
  useGetFlag: (flag: string) => flag !== "autogpt-new-layout",
  useFlagStatus: () => setFlagStatusMock(),
}));

// Uploads go straight to the backend (not through the MSW-mocked proxy), so
// the direct-upload helper is stubbed here.
vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: uploadFileDirectMock,
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

function useFilesHandler(files: WorkspaceFileItem[]) {
  server.use(
    getListWorkspaceFilesMockHandler({
      files,
      offset: 0,
      has_more: false,
    }),
  );
}

// The row's name button is the tooltip trigger; focusing it opens the large
// preview without waiting for the hover delay.
async function openHoverPreview() {
  fireEvent.focus(await screen.findByTestId("artifacts-card-open"));
}

describe("ArtifactsPage - basic rendering", () => {
  test("renders the page header", async () => {
    useStorageHandler();
    useFilesHandler([]);

    render(<ArtifactsPage />);

    expect(
      await screen.findByRole("heading", { name: /files/i }),
    ).toBeDefined();
  });

  test("shows the empty state when the workspace has no files", async () => {
    useStorageHandler();
    useFilesHandler([]);
    server.use(getListWorkspaceFoldersMockHandler({ folders: [] }));

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("artifacts-empty")).toBeDefined();
    expect(screen.getByText(/no files yet/i)).toBeDefined();
  });

  test("shows a quiet hint instead of the empty state when only folders exist", async () => {
    useStorageHandler();
    useFilesHandler([]);
    server.use(
      getListWorkspaceFoldersMockHandler({
        folders: [
          {
            id: "fld-1",
            workspace_id: "ws-1",
            name: "Reports",
            file_count: 2,
            created_at: "2026-05-01T00:00:00Z" as unknown as Date,
            updated_at: "2026-05-01T00:00:00Z" as unknown as Date,
          },
        ],
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("workspace-folder")).toBeDefined();
    expect(await screen.findByText(/no files at the root yet/i)).toBeDefined();
    expect(screen.queryByText(/^no files yet$/i)).toBeNull();
  });

  test("waits for folders before choosing an empty state", async () => {
    useStorageHandler();
    let filesServed = false;
    // Hold the folders response until the test releases it, so the
    // "files empty, folders unknown" state can be asserted deterministically.
    let releaseFolders = () => {};
    const foldersReady = new Promise<void>((resolve) => {
      releaseFolders = resolve;
    });
    server.use(
      http.get("/api/proxy/api/workspace/files", () => {
        filesServed = true;
        return HttpResponse.json({ files: [], offset: 0, has_more: false });
      }),
      http.get("/api/proxy/api/workspace/folders", async () => {
        await foldersReady;
        return HttpResponse.json({
          folders: [
            {
              id: "fld-1",
              workspace_id: "ws-1",
              name: "Reports",
              file_count: 2,
              created_at: "2026-05-01T00:00:00Z",
              updated_at: "2026-05-01T00:00:00Z",
            },
          ],
        });
      }),
    );

    render(<ArtifactsPage />);

    await waitFor(() => expect(filesServed).toBe(true));
    // Files are known to be empty, but folders are still loading: no empty
    // state yet, only skeleton rows.
    expect(screen.getByTestId("artifacts-loading")).toBeDefined();
    expect(screen.queryByTestId("artifacts-empty")).toBeNull();

    releaseFolders();

    expect(await screen.findByText(/no files at the root yet/i)).toBeDefined();
  });

  test("renders one row per file with name, date and size columns", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "f1",
        name: "report.pdf",
        mime_type: "application/pdf",
        size_bytes: 2_516_582,
      }),
      makeFile({ id: "f2", name: "data.csv", mime_type: "text/csv" }),
    ]);

    render(<ArtifactsPage />);

    expect(await screen.findByText("report.pdf")).toBeDefined();
    expect(screen.getByText("data.csv")).toBeDefined();
    expect(screen.getAllByTestId("artifacts-list-item").length).toBe(2);
    expect(screen.getByText("Modified")).toBeDefined();
    expect(screen.getByText("2.4 MB")).toBeDefined();
  });

  test("renders the error card when the API fails", async () => {
    useStorageHandler();
    server.use(getListWorkspaceFilesMockHandler401());

    render(<ArtifactsPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
  });
});

describe("ArtifactsPage - layout toggle", () => {
  test("defaults to the list and switches to the card grid", async () => {
    useStorageHandler();
    useFilesHandler([makeFile({ id: "f1", name: "alpha.txt" })]);

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("artifacts-table")).toBeDefined();

    fireEvent.click(screen.getByTestId("artifacts-view-grid"));

    expect(await screen.findByTestId("artifacts-grid")).toBeDefined();
    expect(screen.queryByTestId("artifacts-table")).toBeNull();
    expect(screen.getByText("alpha.txt")).toBeDefined();
  });
});

describe("ArtifactsPage - search filter", () => {
  test("typing in the search bar narrows the visible rows", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({ id: "f1", name: "alpha.txt" }),
      makeFile({ id: "f2", name: "beta.txt" }),
    ]);

    render(<ArtifactsPage />);

    expect(await screen.findByText("alpha.txt")).toBeDefined();
    expect(screen.getByText("beta.txt")).toBeDefined();

    const search = screen.getByPlaceholderText(/search/i);

    // Second handler returns only beta — the debounce + refetch should show it.
    useFilesHandler([makeFile({ id: "f2", name: "beta.txt" })]);

    fireEvent.change(search, { target: { value: "beta" } });

    // Wait for the filtered list to appear (debounced ~250ms).
    expect(await screen.findByText("beta.txt")).toBeDefined();
  });

  test("searching at root spans folders instead of forcing root_only", async () => {
    useStorageHandler();
    const rootOnlyParams: (string | null)[] = [];
    server.use(
      http.get("/api/proxy/api/workspace/files", ({ request }) => {
        rootOnlyParams.push(new URL(request.url).searchParams.get("root_only"));
        return HttpResponse.json({ files: [], offset: 0, has_more: false });
      }),
    );

    render(<ArtifactsPage />);

    // Initial root listing is scoped to root-level files.
    await waitFor(() => expect(rootOnlyParams).toContain("true"));

    const search = screen.getByPlaceholderText(/search/i);
    fireEvent.change(search, { target: { value: "beta" } });

    // A global search must not be limited to root — files inside folders count.
    await waitFor(() => {
      expect(rootOnlyParams[rootOnlyParams.length - 1]).toBe("false");
    });
  });
});

describe("ArtifactsPage - feature flag gating", () => {
  test("shows the flag-loading skeleton while LaunchDarkly is resolving", async () => {
    setFlagStatusMock.mockReturnValue({ enabled: false, ready: false });

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

describe("ArtifactsPage - new menu", () => {
  test("offers upload and new-folder actions", async () => {
    useStorageHandler();
    useFilesHandler([]);

    render(<ArtifactsPage />);

    await screen.findByTestId("artifacts-empty");
    // Radix DropdownMenu opens on pointerdown, not click, under happy-dom.
    fireEvent.pointerDown(screen.getByTestId("artifacts-new-menu"), {
      button: 0,
    });

    expect(await screen.findByTestId("artifacts-upload-file")).toBeDefined();
    expect(screen.getByTestId("create-folder-button")).toBeDefined();
  });

  test("uploading a file posts it and refreshes the list", async () => {
    useStorageHandler();
    uploadFileDirectMock.mockResolvedValue({
      file_id: "new-1",
      name: "up.txt",
      path: "/up.txt",
      mime_type: "text/plain",
      size_bytes: 3,
    });
    let listCalls = 0;
    server.use(
      http.get("/api/proxy/api/workspace/files", () => {
        listCalls += 1;
        return HttpResponse.json({ files: [], offset: 0, has_more: false });
      }),
    );

    render(<ArtifactsPage />);

    await screen.findByTestId("artifacts-empty");
    const file = new File(["abc"], "up.txt", { type: "text/plain" });
    fireEvent.change(screen.getByTestId("artifacts-upload-input"), {
      target: { files: [file] },
    });

    await waitFor(() =>
      expect(uploadFileDirectMock).toHaveBeenCalledWith(file, undefined, {
        organizationId: null,
        teamId: null,
      }),
    );
    await waitFor(() => expect(listCalls).toBeGreaterThan(1));
  });

  test("uploading inside a folder moves the new file into it", async () => {
    useStorageHandler();
    uploadFileDirectMock.mockResolvedValue({
      file_id: "new-1",
      name: "up.txt",
      path: "/up.txt",
      mime_type: "text/plain",
      size_bytes: 3,
    });
    let movedTo: string | null | undefined;
    server.use(
      http.get("/api/proxy/api/workspace/folders", () =>
        HttpResponse.json({
          folders: [
            {
              id: "fld-1",
              workspace_id: "ws-1",
              name: "Reports",
              file_count: 0,
              created_at: "2026-05-01T00:00:00Z",
              updated_at: "2026-05-01T00:00:00Z",
            },
          ],
        }),
      ),
      http.get("/api/proxy/api/workspace/files", () =>
        HttpResponse.json({ files: [], offset: 0, has_more: false }),
      ),
      http.post(
        "/api/proxy/api/workspace/folders/files/bulk-move",
        async ({ request }) => {
          const body = (await request.json()) as {
            file_ids: string[];
            folder_id: string | null;
          };
          movedTo = body.folder_id;
          return HttpResponse.json([]);
        },
      ),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("workspace-folder"));
    await screen.findByTestId("folder-breadcrumb");

    fireEvent.change(screen.getByTestId("artifacts-upload-input"), {
      target: { files: [new File(["abc"], "up.txt", { type: "text/plain" })] },
    });

    await waitFor(() => expect(movedTo).toBe("fld-1"));
  });
});

describe("ArtifactsPage - row menu", () => {
  test("renders the row actions menu trigger", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "with-menu",
        name: "menu-target.txt",
        path: "/sessions/sess-xyz/menu-target.txt",
      }),
    ]);

    render(<ArtifactsPage />);

    expect(await screen.findByText("menu-target.txt")).toBeDefined();
    expect(screen.getByTestId("artifacts-card-menu")).toBeDefined();
  });

  test("clicking the row opens the file viewer modal", async () => {
    useStorageHandler();
    useFilesHandler([makeFile({ id: "open-me", name: "open-me.txt" })]);
    server.use(
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

describe("ArtifactsPage - previews", () => {
  test("image rows use the file tenant for both preview sizes", async () => {
    const previews: {
      width: string | null;
      org: string | null;
      team: string | null;
    }[] = [];
    server.use(
      http.get("/api/proxy/api/workspace/files/img1/preview", ({ request }) => {
        previews.push({
          width: new URL(request.url).searchParams.get("w"),
          org: request.headers.get("X-Org-Id"),
          team: request.headers.get("X-Team-Id"),
        });
        return new HttpResponse("image", {
          headers: { "Content-Type": "image/png" },
        });
      }),
    );
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "img1",
        name: "pic.png",
        mime_type: "image/png",
        organization_id: "org-file",
        team_id: "team-file",
      }),
    ]);

    render(<ArtifactsPage />);

    await screen.findByText("pic.png");
    await waitFor(() =>
      expect(
        screen
          .getByTestId("artifacts-thumbnail")
          .querySelector("img")
          ?.getAttribute("src"),
      ).toMatch(/^blob:/),
    );

    await openHoverPreview();

    const preview = (await screen.findByAltText("pic.png")) as HTMLImageElement;
    expect(preview.getAttribute("src")).toMatch(/^blob:/);
    expect(screen.getByTestId("artifacts-preview-card")).toBeDefined();
    expect(previews).toEqual(
      expect.arrayContaining([
        { width: "96", org: "org-file", team: "team-file" },
        { width: "800", org: "org-file", team: "team-file" },
      ]),
    );
  });

  test("pdf rows preview through the image endpoint", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "pdf1",
        name: "report.pdf",
        mime_type: "application/pdf",
      }),
    ]);

    render(<ArtifactsPage />);

    await openHoverPreview();

    const img = (await screen.findByAltText("report.pdf")) as HTMLImageElement;
    expect(img.getAttribute("src")).toMatch(/^blob:/);
  });

  test("text rows fall back to a type icon instead of a thumbnail", async () => {
    useStorageHandler();
    useFilesHandler([makeFile({ id: "txt1", name: "notes.txt" })]);

    render(<ArtifactsPage />);

    await screen.findByText("notes.txt");
    expect(
      screen.getByTestId("artifacts-thumbnail").querySelector("img"),
    ).toBeNull();
  });

  test("csv previews render a table from the byte-capped preview", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({ id: "csv1", name: "data.csv", mime_type: "text/csv" }),
    ]);
    server.use(
      http.get("/api/proxy/api/workspace/files/csv1/preview", () =>
        HttpResponse.text("name,age\nAda,36\nBob,40\n"),
      ),
    );

    render(<ArtifactsPage />);

    await openHoverPreview();

    expect(await screen.findByText("name")).toBeDefined();
    expect(await screen.findByText("Ada")).toBeDefined();
  });

  test("ics previews render the event summary", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "ics1",
        name: "meeting.ics",
        mime_type: "text/calendar",
        size_bytes: 400,
      }),
    ]);
    server.use(
      http.get("/api/proxy/api/workspace/files/ics1/preview", () =>
        HttpResponse.text(
          "BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Launch sync\nDTSTART:20260615T130000Z\nLOCATION:Room 4\nEND:VEVENT\nEND:VCALENDAR",
        ),
      ),
    );

    render(<ArtifactsPage />);

    await openHoverPreview();

    expect(await screen.findByText("Launch sync")).toBeDefined();
  });

  test("vcard previews render the contact name", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "vcf1",
        name: "ada.vcf",
        mime_type: "text/vcard",
        size_bytes: 300,
      }),
    ]);
    server.use(
      http.get("/api/proxy/api/workspace/files/vcf1/preview", () =>
        HttpResponse.text(
          "BEGIN:VCARD\nFN:Ada Lovelace\nORG:Analytical Engine\nEND:VCARD",
        ),
      ),
    );

    render(<ArtifactsPage />);

    await openHoverPreview();

    expect(await screen.findByText("Ada Lovelace")).toBeDefined();
  });

  test("markdown previews render their formatted content", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({ id: "md1", name: "notes.md", mime_type: "text/markdown" }),
    ]);
    server.use(
      http.get("/api/proxy/api/workspace/files/md1/preview", () =>
        HttpResponse.text("# Heading One\n\nbody paragraph"),
      ),
    );

    render(<ArtifactsPage />);

    await openHoverPreview();

    expect(await screen.findByText("Heading One")).toBeDefined();
    expect(await screen.findByText("body paragraph")).toBeDefined();
  });

  test("grid cards request the card-sized preview", async () => {
    const widths: (string | null)[] = [];
    server.use(
      http.get("/api/proxy/api/workspace/files/img1/preview", ({ request }) => {
        widths.push(new URL(request.url).searchParams.get("w"));
        return new HttpResponse("image");
      }),
    );
    useStorageHandler();
    useFilesHandler([
      makeFile({ id: "img1", name: "pic.png", mime_type: "image/png" }),
    ]);

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-view-grid"));

    const img = (await screen.findByAltText("pic.png")) as HTMLImageElement;
    expect(img.getAttribute("src")).toMatch(/^blob:/);
    expect(widths).toContain("400");
  });
});

describe("ArtifactsPage - file viewer modal", () => {
  test("opening a markdown file shows a Source toggle that flips to Preview", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({ id: "md2", name: "doc.md", mime_type: "text/markdown" }),
    ]);
    server.use(
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
    useFilesHandler([
      makeFile({
        id: "zip1",
        name: "archive.zip",
        mime_type: "application/zip",
      }),
    ]);

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));

    expect(await screen.findByText(/can't be previewed/i)).toBeDefined();
  });

  test("opening an uploaded file resolves its source ref", async () => {
    useStorageHandler();
    useFilesHandler([
      makeFile({
        id: "up1",
        name: "uploaded.md",
        mime_type: "text/markdown",
        origin: "uploaded",
      }),
    ]);
    server.use(
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
    useFilesHandler([
      makeFile({ id: "dl1", name: "doc.md", mime_type: "text/markdown" }),
    ]);
    server.use(
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

test("leaves the selected folder and uploads to the new organization after a switch", async () => {
  useStorageHandler();
  useOrgTeamStore.setState({ activeOrgID: "org-one", activeTeamID: null });
  const listings: Array<{ org: string | null; folder: string | null }> = [];
  server.use(
    getListWorkspaceFoldersMockHandler(({ request }) => ({
      folders:
        request.headers.get("X-Org-Id") === "org-one"
          ? [
              {
                id: "folder-one",
                workspace_id: "workspace-one",
                name: "Team reports",
                organization_id: "org-one",
                team_id: "team-one",
                created_at: new Date(),
                updated_at: new Date(),
              },
            ]
          : [],
    })),
    getListWorkspaceFilesMockHandler(({ request }) => {
      const org = request.headers.get("X-Org-Id");
      const folder = new URL(request.url).searchParams.get("folder_id");
      listings.push({ org, folder });
      return { files: [], offset: 0, has_more: false };
    }),
  );
  uploadFileDirectMock.mockResolvedValue({
    file_id: "new-file",
    name: "new.txt",
    path: "/new.txt",
    mime_type: "text/plain",
    size_bytes: 3,
  });
  render(<ArtifactsPage />);
  fireEvent.click(await screen.findByTestId("workspace-folder"));
  await waitFor(() =>
    expect(listings).toContainEqual({ org: "org-one", folder: "folder-one" }),
  );
  act(() =>
    useOrgTeamStore.setState({ activeOrgID: "org-two", activeTeamID: null }),
  );
  await waitFor(() =>
    expect(listings).toContainEqual({ org: "org-two", folder: null }),
  );
  expect(screen.queryByTestId("folder-breadcrumb")).toBeNull();
  fireEvent.change(screen.getByTestId("artifacts-upload-input"), {
    target: { files: [new File(["abc"], "new.txt", { type: "text/plain" })] },
  });
  await waitFor(() =>
    expect(uploadFileDirectMock).toHaveBeenCalledWith(
      expect.any(File),
      undefined,
      {
        organizationId: "org-two",
        teamId: null,
      },
    ),
  );
});

test("filters Files, folders, storage, and new content to the selected member team", async () => {
  useOrgTeamStore.setState({
    activeOrgID: "org-one",
    activeTeamID: null,
    teams: [
      {
        id: "team-one",
        orgId: "org-one",
        name: "QA team",
        slug: "qa",
        isDefault: false,
        joinPolicy: "INVITE_ONLY",
      },
      {
        id: "foreign-team",
        orgId: "other-org",
        name: "Foreign team",
        slug: "foreign",
        isDefault: false,
        joinPolicy: "INVITE_ONLY",
      },
    ],
  });
  const storageScopes: Array<string | null> = [];
  let createdFolderScope: { org: string | null; team: string | null } | null =
    null;
  server.use(
    http.post("/api/proxy/api/workspace/folders", async ({ request }) => {
      createdFolderScope = {
        org: request.headers.get("X-Org-Id"),
        team: request.headers.get("X-Team-Id"),
      };
      return HttpResponse.json({
        id: "new-folder",
        ...((await request.json()) as { name: string }),
      });
    }),
    getListWorkspaceFoldersMockHandler(({ request }) => ({
      folders:
        request.headers.get("X-Team-Id") === "team-one"
          ? [
              {
                id: "team-folder",
                workspace_id: "workspace-one",
                name: "Team reports",
                organization_id: "org-one",
                team_id: "team-one",
                created_at: new Date(),
                updated_at: new Date(),
              },
            ]
          : [],
    })),
    getListWorkspaceFilesMockHandler(({ request }) => ({
      files: [
        makeFile({
          id: request.headers.get("X-Team-Id") ?? "home",
          name:
            request.headers.get("X-Team-Id") === "team-one"
              ? "team-only.txt"
              : "org-home.txt",
        }),
      ],
      offset: 0,
      has_more: false,
    })),
    getGetWorkspaceStorageUsageMockHandler(({ request }) => {
      storageScopes.push(request.headers.get("X-Team-Id"));
      return {
        used_bytes: 0,
        limit_bytes: 1000,
        used_percent: 0,
        file_count: 0,
      };
    }),
  );
  uploadFileDirectMock.mockResolvedValue({
    file_id: "new-file",
    name: "new.txt",
    path: "/new.txt",
    mime_type: "text/plain",
    size_bytes: 3,
  });
  render(<ArtifactsPage />);
  expect(await screen.findByText("org-home.txt")).toBeDefined();
  fireEvent.click(screen.getByRole("combobox", { name: "Files in" }));
  expect(screen.queryByRole("option", { name: "Foreign team" })).toBeNull();
  fireEvent.click(await screen.findByRole("option", { name: "QA team" }));
  expect(await screen.findByText("team-only.txt")).toBeDefined();
  expect(screen.queryByText("org-home.txt")).toBeNull();
  expect(await screen.findByText("Team reports")).toBeDefined();
  await waitFor(() => expect(storageScopes).toContain("team-one"));
  fireEvent.pointerDown(screen.getByTestId("artifacts-new-menu"), {
    button: 0,
  });
  fireEvent.click(await screen.findByTestId("create-folder-button"));
  fireEvent.change(await screen.findByLabelText("Folder name"), {
    target: { value: "New team folder" },
  });
  fireEvent.click(screen.getByTestId("folder-form-submit"));
  await waitFor(() =>
    expect(createdFolderScope).toEqual({ org: "org-one", team: "team-one" }),
  );
  await waitFor(() =>
    expect(screen.queryByTestId("folder-form-submit")).toBeNull(),
  );
  fireEvent.change(screen.getByTestId("artifacts-upload-input"), {
    target: { files: [new File(["abc"], "new.txt", { type: "text/plain" })] },
  });
  await waitFor(() =>
    expect(uploadFileDirectMock).toHaveBeenCalledWith(
      expect.any(File),
      undefined,
      { organizationId: "org-one", teamId: "team-one" },
    ),
  );
  fireEvent.click(screen.getByRole("combobox", { name: "Files in" }));
  fireEvent.click(await screen.findByRole("option", { name: "Organization" }));
  expect(await screen.findByText("org-home.txt")).toBeDefined();
  expect(screen.queryByText("team-only.txt")).toBeNull();
  expect(screen.queryByText("Team reports")).toBeNull();
});
