import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { WorkspaceFilePicker } from "../WorkspaceFilePicker";

const mockListWorkspaceFiles = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  listWorkspaceFiles: (...args: unknown[]) => mockListWorkspaceFiles(...args),
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

    expect(onConfirm).toHaveBeenCalledWith([FILE]);
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

    expect(onConfirm).toHaveBeenCalledWith([files[1], files[2], files[3]]);
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
