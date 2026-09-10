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
});
