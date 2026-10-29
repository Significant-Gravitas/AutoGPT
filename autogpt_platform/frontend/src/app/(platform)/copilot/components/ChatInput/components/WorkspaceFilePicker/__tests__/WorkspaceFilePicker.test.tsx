import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkspaceFilePicker } from "../WorkspaceFilePicker";

const mockListWorkspaceFiles = vi.fn();
const mockFolders = vi.fn<() => { data?: { folders: unknown[] } }>(() => ({
  data: { folders: [] },
}));
vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  listWorkspaceFiles: (...args: unknown[]) => mockListWorkspaceFiles(...args),
  useListWorkspaceFolders: () => mockFolders(),
}));

const FILE = {
  id: "file-1",
  name: "alpha.txt",
  path: "/workspace/alpha.txt",
  mime_type: "text/plain",
  size_bytes: 10,
  created_at: "2026-01-01T00:00:00Z",
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("WorkspaceFilePicker", () => {
  it("scopes the listing to the expert the chat is addressed to", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
        expertId="expert-a"
      />,
    );

    await screen.findByText("alpha.txt");
    expect(mockListWorkspaceFiles).toHaveBeenCalledWith(
      expect.objectContaining({ expert_id: "expert-a" }),
    );
  });

  it("drops the previous expert's files while the next expert's listing is pending", async () => {
    mockListWorkspaceFiles.mockResolvedValueOnce({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const { rerender } = render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
        expertId="expert-a"
      />,
    );
    await screen.findByText("alpha.txt");

    // Expert B's request never settles within the test: A's file must not
    // stay on screen as a placeholder in the meantime.
    mockListWorkspaceFiles.mockReturnValue(new Promise(() => {}));
    rerender(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
        expertId="expert-b"
      />,
    );

    await waitFor(() => expect(screen.queryByText("alpha.txt")).toBeNull());
    expect(mockListWorkspaceFiles).toHaveBeenLastCalledWith(
      expect.objectContaining({ expert_id: "expert-b" }),
    );
  });

  it("lists the whole workspace for a personal chat", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    await screen.findByText("alpha.txt");
    expect(mockListWorkspaceFiles.mock.calls[0][0].expert_id).toBeUndefined();
  });

  it("lists workspace files and confirms the selection", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const onConfirm = vi.fn();
    const onClose = vi.fn();

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={onClose}
        onConfirm={onConfirm}
      />,
    );

    // File appears once the query resolves.
    fireEvent.click(await screen.findByText("alpha.txt"));

    fireEvent.click(await screen.findByRole("button", { name: /add 1 file/i }));

    expect(onConfirm).toHaveBeenCalledWith([{ kind: "file", file: FILE }]);
    expect(onClose).toHaveBeenCalled();
  });

  it("disables Add until a file is selected", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    await screen.findByText("alpha.txt");
    expect(screen.getByRole("button", { name: /^add$/i })).toHaveProperty(
      "disabled",
      true,
    );
  });

  it("shows an empty state when the workspace has no files", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [], has_more: false },
    });

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    expect(
      await screen.findByText(/no files in your workspace yet/i),
    ).toBeTruthy();
  });

  it("offers a Load more action when there are more pages", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: true },
    });

    render(
      <WorkspaceFilePicker
        isOpen={true}
        onClose={vi.fn()}
        onConfirm={vi.fn()}
      />,
    );

    await screen.findByText("alpha.txt");
    expect(
      await screen.findByRole("button", { name: /load more/i }),
    ).toBeTruthy();
  });

  it("shift-click selects the inclusive range from the last plain click", async () => {
    mockPage(makeFiles(5));
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={vi.fn()} />,
    );
    await screen.findByText("f-0");

    fireEvent.click(card("f-1"));
    fireEvent.click(card("f-3"), { shiftKey: true });

    expect(selectedNames()).toEqual(["f-1", "f-2", "f-3"]);
  });

  it("ranges backwards too, and confirms the whole range", async () => {
    const files = makeFiles(5);
    mockPage(files);
    const onConfirm = vi.fn();
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={onConfirm} />,
    );
    await screen.findByText("f-0");

    fireEvent.click(card("f-3"));
    fireEvent.click(card("f-1"), { shiftKey: true });
    fireEvent.click(screen.getByRole("button", { name: /add 3 file/i }));

    expect(onConfirm).toHaveBeenCalledWith(
      [files[1], files[2], files[3]].map((file) => ({ kind: "file", file })),
    );
  });

  // Firefox reports shiftKey false on the click it synthesises from
  // Shift+Enter, so the chord is handled on keydown instead.
  it("ranges from the anchor on Shift+Enter and Shift+Space", async () => {
    mockPage(makeFiles(5));
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={vi.fn()} />,
    );
    await screen.findByText("f-0");

    fireEvent.click(card("f-0"));
    fireEvent.keyDown(card("f-2"), { key: "Enter", shiftKey: true });
    expect(selectedNames()).toEqual(["f-0", "f-1", "f-2"]);

    fireEvent.keyDown(card("f-4"), { key: " ", shiftKey: true });
    expect(selectedNames()).toEqual(["f-0", "f-1", "f-2", "f-3", "f-4"]);
  });

  it("leaves a plain Enter to the button's own click", async () => {
    mockPage(makeFiles(3));
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={vi.fn()} />,
    );
    await screen.findByText("f-0");

    fireEvent.keyDown(card("f-1"), { key: "Enter" });

    expect(selectedNames()).toEqual([]);
  });

  it("drops the anchor when the filter changes", async () => {
    mockPage(makeFiles(5));
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={vi.fn()} />,
    );
    await screen.findByText("f-0");
    fireEvent.click(card("f-1"));

    fireEvent.change(screen.getByLabelText(/search workspace files/i), {
      target: { value: "f" },
    });
    fireEvent.click(card("f-3"), { shiftKey: true });

    // Without an anchor the Shift+click is a plain toggle, not a range.
    expect(selectedNames()).toEqual(["f-1", "f-3"]);
  });

  // No listbox role: that promises arrow-key navigation the grid lacks.
  it("exposes each card's selection as a pressed toggle button", async () => {
    mockPage(makeFiles(2));
    render(
      <WorkspaceFilePicker isOpen onClose={vi.fn()} onConfirm={vi.fn()} />,
    );
    await screen.findByText("f-0");

    expect(screen.queryByRole("listbox")).toBeNull();
    expect(card("f-0").getAttribute("aria-pressed")).toBe("false");

    fireEvent.click(card("f-0"));
    expect(card("f-0").getAttribute("aria-pressed")).toBe("true");
  });
});

const FOLDERS = [
  {
    id: "fld-1",
    workspace_id: "ws-1",
    name: "Reports",
    parent_id: null,
    file_count: 3,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
  {
    id: "fld-2",
    workspace_id: "ws-1",
    name: "2026",
    parent_id: "fld-1",
    file_count: 0,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  },
];

function renderPicker(
  props: Partial<{ expertId: string; expertName: string }> = {},
) {
  const onConfirm = vi.fn();
  render(
    <WorkspaceFilePicker
      isOpen={true}
      onClose={vi.fn()}
      onConfirm={onConfirm}
      expertId={props.expertId ?? null}
      expertName={props.expertName ?? null}
    />,
  );
  return { onConfirm };
}

function lastRequest() {
  return mockListWorkspaceFiles.mock.calls.at(-1)?.[0] as Record<
    string,
    unknown
  >;
}

describe("WorkspaceFilePicker - the expert-only filter", () => {
  it("opens ON in an expert chat and asks for that expert's conversations alone", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker({ expertId: "expert-a", expertName: "Maria" });

    await screen.findByText("alpha.txt");
    expect(screen.getByLabelText("Only Maria’s files")).toBeTruthy();
    expect(lastRequest()).toMatchObject({
      expert_id: "expert-a",
      include_user_files: false,
    });
    // No folder axis while the filter is on: the user's folders are exactly
    // what it hides.
    expect(lastRequest().root_only).toBeUndefined();
    expect(screen.queryByRole("list", { name: "Folders" })).toBeNull();
  });

  it("switching it off widens to the user's own files and shows folder rows", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker({ expertId: "expert-a", expertName: "Maria" });
    await screen.findByText("alpha.txt");

    fireEvent.click(screen.getByLabelText("Only Maria’s files"));

    await waitFor(() =>
      expect(lastRequest()).toMatchObject({
        expert_id: "expert-a",
        include_user_files: true,
        root_only: true,
      }),
    );
    expect(await screen.findByRole("list", { name: "Folders" })).toBeTruthy();
    expect(screen.getByText("Plus your own files and folders")).toBeTruthy();
  });

  it("renders no switch in a personal chat, and shows folders straight away", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker();

    await screen.findByText("alpha.txt");
    expect(screen.queryByLabelText(/Only .*files/)).toBeNull();
    expect(screen.getByRole("list", { name: "Folders" })).toBeTruthy();
    expect(lastRequest().include_user_files).toBeUndefined();
  });

  it("the ON empty state offers to widen, and the button flips the switch", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker({ expertId: "expert-a", expertName: "Maria" });

    expect(await screen.findByText("No files from Maria yet.")).toBeTruthy();
    fireEvent.click(
      screen.getByRole("button", { name: /show your own files too/i }),
    );

    await waitFor(() =>
      expect(lastRequest()).toMatchObject({ include_user_files: true }),
    );
  });
});

describe("WorkspaceFilePicker - folders", () => {
  it("lists the root's folders and navigates into one", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker();
    await screen.findByText("alpha.txt");

    // Only root folders at the root — "2026" is Reports' child.
    expect(screen.getByText("Reports")).toBeTruthy();
    expect(screen.queryByText("2026")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Open Reports" }));

    await waitFor(() =>
      expect(lastRequest()).toMatchObject({ folder_id: "fld-1" }),
    );
    expect(await screen.findByText("2026")).toBeTruthy();
    expect(screen.getByTestId("folder-breadcrumb")).toBeTruthy();

    fireEvent.click(screen.getByTestId("folder-breadcrumb-root"));
    await waitFor(() =>
      expect(lastRequest()).toMatchObject({ root_only: true }),
    );
  });

  it("attaching a folder selects it without opening it, and hands it to the composer", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    const { onConfirm } = renderPicker();
    await screen.findByText("alpha.txt");

    const attach = screen.getByRole("button", {
      name: "Attach folder Reports",
    });
    fireEvent.click(attach);

    expect(attach.getAttribute("aria-pressed")).toBe("true");
    // Attaching must not navigate: the file listing is still the root's.
    expect(lastRequest().folder_id).toBeUndefined();
    expect(screen.getByRole("button", { name: "Add 1 folder" })).toBeTruthy();

    fireEvent.click(screen.getByText("alpha.txt"));
    expect(
      screen.getByRole("button", { name: "Add 1 file and 1 folder" }),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /^Add / }));
    expect(onConfirm).toHaveBeenCalledWith([
      { kind: "folder", folder: FOLDERS[0], subfolderCount: 1 },
      { kind: "file", file: FILE },
    ]);
  });

  it("opening a folder never shows the previous folder's files while its own load", async () => {
    mockPage(makeFiles(1));
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker();
    await screen.findByText("f-0");

    mockListWorkspaceFiles.mockReturnValue(new Promise(() => {}));
    fireEvent.click(screen.getByRole("button", { name: "Open Reports" }));
    await waitFor(() =>
      expect(lastRequest()).toMatchObject({ folder_id: "fld-1" }),
    );

    expect(screen.queryByText("f-0")).toBeNull();
  });

  it("opening a folder drops the range anchor taken in the previous listing", async () => {
    mockPage(makeFiles(3));
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker();
    await screen.findByText("f-0");

    fireEvent.click(card("f-0"));
    fireEvent.click(screen.getByRole("button", { name: "Open Reports" }));
    await waitFor(() =>
      expect(lastRequest()).toMatchObject({ folder_id: "fld-1" }),
    );
    await screen.findByText("f-2");
    fireEvent.click(card("f-2"), { shiftKey: true });

    expect(selectedNames()).toEqual(["f-0", "f-2"]);
  });

  it("a search spans every folder and hides the folder rows", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    mockFolders.mockReturnValue({ data: { folders: FOLDERS } });

    renderPicker();
    await screen.findByText("alpha.txt");

    fireEvent.change(screen.getByLabelText("Search workspace files"), {
      target: { value: "alpha" },
    });

    await waitFor(() => expect(lastRequest().q).toBe("alpha"));
    expect(lastRequest().root_only).toBeUndefined();
    expect(screen.queryByRole("list", { name: "Folders" })).toBeNull();
  });
});

function mockPage(files: ReturnType<typeof makeFiles>) {
  mockListWorkspaceFiles.mockResolvedValue({
    status: 200,
    data: { files, has_more: false },
  });
}

function card(name: string) {
  return screen.getByRole("button", { name: new RegExp(`^${name}\\b`) });
}

function selectedNames() {
  return screen
    .getAllByRole("button")
    .filter((el) => el.getAttribute("aria-pressed") === "true")
    .map((el) => el.querySelector("[title]")?.textContent ?? "")
    .sort();
}

function makeFiles(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    ...FILE,
    id: `id-${i}`,
    name: `f-${i}`,
    path: `/workspace/f-${i}`,
  }));
}
