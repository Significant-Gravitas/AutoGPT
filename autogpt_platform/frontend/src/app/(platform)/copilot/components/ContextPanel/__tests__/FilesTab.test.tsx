import { describe, expect, test } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getDeleteWorkspaceFileMockHandler200,
  getListWorkspaceFilesMockHandler200,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { FilesTab } from "../components/FilesTab/FilesTab";

const SESSION = "session-1";

function listResponse(): ListFilesResponse {
  return {
    files: [
      {
        id: "aaaaaaaa-0000-0000-0000-000000000001",
        name: "uploaded.png",
        path: "/sessions/session-1/uploaded.png",
        mime_type: "image/png",
        size_bytes: 1024,
        metadata: { origin: "user-upload" },
        origin: "uploaded",
        created_at: "2026-05-20T10:00:00Z",
      },
      {
        id: "bbbbbbbb-0000-0000-0000-000000000002",
        name: "result.csv",
        path: "/sessions/session-1/result.csv",
        mime_type: "text/csv",
        size_bytes: 4096,
        metadata: { origin: "agent" },
        origin: "generated",
        created_at: "2026-05-20T11:00:00Z",
      },
    ],
    offset: 0,
    has_more: false,
  };
}

describe("FilesTab", () => {
  test("renders Uploaded and Generated sections from the API", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse()));
    render(<FilesTab sessionId={SESSION} />);
    expect(await screen.findByText("uploaded.png")).toBeDefined();
    expect(await screen.findByText("result.csv")).toBeDefined();
    expect(screen.getByRole("heading", { name: /Uploaded/i })).toBeDefined();
    expect(screen.getByRole("heading", { name: /Generated/i })).toBeDefined();
  });

  test("shows empty state when the session has no files", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200({
        files: [],
        offset: 0,
        has_more: false,
      }),
    );
    render(<FilesTab sessionId={SESSION} />);
    expect(await screen.findByText(/No files yet/i)).toBeDefined();
  });

  test("delete is offered for generated files only and confirms via dialog", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse()));
    server.use(getDeleteWorkspaceFileMockHandler200());
    render(<FilesTab sessionId={SESSION} />);
    await screen.findByText("result.csv");
    expect(screen.queryByLabelText("Delete uploaded.png")).toBeNull();
    fireEvent.click(screen.getByLabelText("Delete result.csv"));
    const dialog = await screen.findByRole("dialog");
    const confirm = within(dialog).getByRole("button", { name: /^Delete$/ });
    fireEvent.click(confirm);
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });
});
