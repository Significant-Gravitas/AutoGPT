import { beforeEach, describe, expect, test, vi } from "vitest";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { MoveToFolderDialog } from "../MoveToFolderDialog";

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

const PROXY = "/api/proxy/api/workspace";

// Files / Reports / 2026 / Q3, plus a sibling at the root.
const FOLDERS = [
  folder("fld-1", "Reports", null),
  folder("fld-2", "2026", "fld-1"),
  folder("fld-3", "Q3", "fld-2"),
  folder("fld-9", "Archive", null),
];

beforeEach(() => {
  server.use(
    http.get(`${PROXY}/folders`, async () => {
      // The dialog mounts before the folders land — a cold cache, which is
      // what a lazy `useState` initialiser cannot see.
      await new Promise((resolve) => setTimeout(resolve, 60));
      return HttpResponse.json({ folders: FOLDERS });
    }),
  );
});

describe("MoveToFolderDialog", () => {
  test("opens on the subject's own branch even when the folders arrive late", async () => {
    render(
      <MoveToFolderDialog
        move={{ kind: "folder", folderId: "fld-2" }}
        subject="“2026”"
        isOpen
        setIsOpen={() => {}}
      />,
    );

    // "2026" is two levels down, so it is only visible if both its parent and
    // its own row were expanded once the query settled.
    expect(await screen.findByText("2026")).toBeDefined();
    await waitFor(() => expect(screen.getByText("Q3")).toBeDefined());

    const rows = screen.getAllByTestId("move-to-folder-option");
    const subject = rows.find((r) => r.textContent?.includes("2026"));
    expect(subject?.textContent).toContain("Folder being moved");
    const parent = rows.find((r) => r.textContent?.startsWith("Reports"));
    expect(parent?.textContent).toContain("Current location");
  });

  test("moves a selection spread across folders to the root", async () => {
    let movedTo: unknown = "not called";
    server.use(
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        movedTo = ((await request.json()) as { folder_id: unknown }).folder_id;
        return HttpResponse.json({});
      }),
    );
    // What SelectionBar passes when the selected files sit in different folders.
    render(
      <MoveToFolderDialog
        move={{ kind: "files", fileIds: ["a", "b"], currentFolderId: null }}
        canMoveToRoot
        subject="2 files"
        isOpen
        setIsOpen={() => {}}
      />,
    );

    const root = await screen.findByText("Files (root)");
    await screen.findByText("Reports");
    expect(root.closest("[role=treeitem]")?.textContent).not.toContain(
      "Current location",
    );
    fireEvent.click(root);
    fireEvent.click(screen.getByTestId("confirm-move-to-folder"));
    await waitFor(() => expect(movedTo).toBeNull());
  });

  test("moves to a selected folder after its parent row is collapsed", async () => {
    let movedTo: unknown = "not called";
    server.use(
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        movedTo = ((await request.json()) as { folder_id: unknown }).folder_id;
        return HttpResponse.json({});
      }),
    );
    render(
      <MoveToFolderDialog
        move={{ kind: "files", fileIds: ["file-1"], currentFolderId: "fld-2" }}
        subject="“a.txt”"
        isOpen
        setIsOpen={() => {}}
      />,
    );

    fireEvent.click(await screen.findByText("Q3"));
    fireEvent.click(screen.getByRole("button", { name: "Collapse Reports" }));
    await waitFor(() => expect(screen.queryByText("Q3")).toBeNull());

    const move = screen.getByTestId("confirm-move-to-folder");
    expect(move.hasAttribute("disabled")).toBe(false);
    fireEvent.click(move);
    await waitFor(() => expect(movedTo).toBe("fld-3"));
  });
});

function folder(id: string, name: string, parentId: string | null) {
  return {
    id,
    workspace_id: "ws-1",
    name,
    parent_id: parentId,
    file_count: 0,
    created_at: "2026-09-01T00:00:00Z",
    updated_at: "2026-09-01T00:00:00Z",
  };
}
