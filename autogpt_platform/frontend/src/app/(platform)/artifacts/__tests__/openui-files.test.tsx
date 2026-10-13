import { beforeEach, describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFoldersMockHandler,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import { formView } from "@/lib/openui/__tests__/capability-fixtures";
import { OPENUI_MIME } from "@/lib/openui/file";
import { resetNavigation } from "./navigation-mock";

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

const source = formView(
  'Field("notes", "Notes", "Original", "", false)',
).replace("[form])", '[form], "files-review")');

beforeEach(() => {
  resetNavigation();
  localStorage.clear();
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getListWorkspaceFoldersMockHandler({ folders: [] }),
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: 0,
      limit_bytes: 1_000_000_000,
      used_percent: 0,
      file_count: 1,
    }),
    getListWorkspaceFilesMockHandler({
      files: [
        {
          id: "view-1",
          name: "Trip.openui",
          path: "/sessions/home-chat/Trip.openui",
          mime_type: OPENUI_MIME,
          size_bytes: source.length,
          metadata: {},
          origin: "generated",
          created_at: "2026-10-01T00:00:00Z",
        },
      ],
      offset: 0,
      has_more: false,
    }),
    http.get("/api/proxy/api/workspace/files/view-1/download", () =>
      HttpResponse.text(source),
    ),
  );
});

describe("ArtifactsPage - saved interactive views", () => {
  test("lists a view as an interactive view and opens it editable", async () => {
    server.use(
      http.get("*/api/workspace/files/view-1/view-state", () =>
        HttpResponse.json({
          file_id: "view-1",
          view_id: "files-review",
          source,
          checksum: "c1",
          state: {
            review: {
              notes: { value: "Saved edit", componentType: "Field" },
            },
          },
          revision: 3,
          origin_session_id: "home-chat",
        }),
      ),
    );
    render(<ArtifactsPage />);

    expect(
      (await screen.findAllByText(/Interactive view/)).length,
    ).toBeGreaterThan(0);
    fireEvent.click(await screen.findByTestId("artifacts-card-open"));
    const viewer = await screen.findByTestId("file-viewer");
    const notes = (await within(viewer).findByLabelText(
      "Notes",
    )) as HTMLInputElement;
    expect(notes.value).toBe("Saved edit");
    expect(notes.disabled).toBe(false);
  });

  test("shows an uploaded .openui file without view data as text", async () => {
    server.use(
      http.get("*/api/workspace/files/view-1/view-state", () =>
        HttpResponse.json({ detail: "Not found" }, { status: 404 }),
      ),
    );
    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-card-open"));
    const viewer = await screen.findByTestId("file-viewer");
    expect(
      await within(viewer).findByText(/root = Workspace\("Review"/),
    ).toBeDefined();
    expect(within(viewer).queryByLabelText("Notes")).toBeNull();
  });
});
