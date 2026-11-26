import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import {
  getGetV2ListLibraryFoldersMockHandler,
  getPostV2BulkMoveAgentsMockHandler,
  getPostV2BulkMoveAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MoveToFolderDialog } from "../MoveToFolderDialog";

function makeFolder(id: string, name: string): LibraryFolder {
  return {
    id,
    user_id: "test-user",
    name,
    agent_count: 1,
    subfolder_count: 0,
    color: null,
    icon: null,
    parent_id: null,
    created_at: new Date(),
    updated_at: new Date(),
  };
}

describe("MoveToFolderDialog", () => {
  test("shows My Library (root) when workflow is in the only folder", async () => {
    const movePayloads: Array<{
      agent_ids: string[];
      folder_id?: string | null;
    }> = [];
    server.use(
      getGetV2ListLibraryFoldersMockHandler({
        folders: [makeFolder("folder-only", "Only Folder")],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
      getPostV2BulkMoveAgentsMockHandler(async ({ request }) => {
        const body = (await request.json()) as {
          agent_ids: string[];
          folder_id?: string | null;
        };
        movePayloads.push(body);
        return getPostV2BulkMoveAgentsResponseMock();
      }),
    );

    const setIsOpen = vi.fn();
    render(
      <MoveToFolderDialog
        agentId="agent-1"
        agentName="My Workflow"
        currentFolderId="folder-only"
        isOpen={true}
        setIsOpen={setIsOpen}
      />,
    );

    const rootButton = await screen.findByRole("button", {
      name: /My Library \(root\)/i,
    });
    expect(rootButton).toBeDefined();
    expect(screen.queryByText("No folders found")).toBeNull();

    await userEvent.click(rootButton);
    await waitFor(() => expect(movePayloads.length).toBeGreaterThanOrEqual(1));
    expect(movePayloads[0]).toEqual({
      agent_ids: ["agent-1"],
      folder_id: null,
    });
  });

  test("keeps My Library (root) visible when folder search matches nothing", async () => {
    server.use(
      getGetV2ListLibraryFoldersMockHandler({
        folders: [
          makeFolder("folder-1", "Alpha"),
          makeFolder("folder-2", "Beta"),
        ],
        pagination: {
          total_items: 2,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    );

    render(
      <MoveToFolderDialog
        agentId="agent-1"
        agentName="My Workflow"
        currentFolderId="folder-1"
        isOpen={true}
        setIsOpen={vi.fn()}
      />,
    );

    expect(await screen.findByRole("button", { name: /Beta/i })).toBeDefined();

    await userEvent.type(
      screen.getByPlaceholderText("Search folders..."),
      "zzz",
    );

    expect(screen.queryByRole("button", { name: /Beta/i })).toBeNull();
    expect(
      screen.getByRole("button", { name: /My Library \(root\)/i }),
    ).toBeDefined();
  });

  test("shows No folders found when already at root and no other folders exist", async () => {
    server.use(
      getGetV2ListLibraryFoldersMockHandler({
        folders: [],
        pagination: {
          total_items: 0,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    );

    render(
      <MoveToFolderDialog
        agentId="agent-1"
        agentName="My Workflow"
        currentFolderId={null}
        isOpen={true}
        setIsOpen={vi.fn()}
      />,
    );

    expect(await screen.findByText("No folders found")).toBeDefined();
    expect(
      screen.queryByRole("button", { name: /My Library \(root\)/i }),
    ).toBeNull();
  });
});
