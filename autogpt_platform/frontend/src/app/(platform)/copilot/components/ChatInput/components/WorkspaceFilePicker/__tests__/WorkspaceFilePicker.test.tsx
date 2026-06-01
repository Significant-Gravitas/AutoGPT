import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
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
    expect(screen.getByRole("button", { name: /^add$/i })).toBeDisabled();
  });
});
