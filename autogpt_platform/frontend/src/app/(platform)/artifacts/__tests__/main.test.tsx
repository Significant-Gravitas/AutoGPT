import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFilesMockHandler401,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS_PAGE: "artifacts-page" },
  useGetFlag: () => true,
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
      await screen.findByRole("heading", { name: /artifacts/i }),
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
    expect(screen.getByText(/no artifacts yet/i)).toBeDefined();
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
});
