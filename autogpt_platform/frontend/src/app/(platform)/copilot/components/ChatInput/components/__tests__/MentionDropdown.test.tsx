import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { MutableRefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { MentionDropdown } from "../MentionDropdown";
import type { MentionOption } from "../../useChatMentions";

const FILE: WorkspaceFileItem = {
  id: "file-1",
  name: "alpha.txt",
  path: "/workspace/alpha.txt",
  mime_type: "text/plain",
  size_bytes: 10,
  origin: "uploaded",
  created_at: "2026-01-01T00:00:00Z",
};

function renderDropdown(
  overrides: Partial<Parameters<typeof MentionDropdown>[0]> = {},
) {
  const props = {
    options: [] as MentionOption[],
    isLoading: false,
    isError: false,
    highlightedIndex: 0,
    highlightedRef: {
      current: null,
    } as MutableRefObject<HTMLButtonElement | null>,
    onSelect: vi.fn(),
    onHighlight: vi.fn(),
    ...overrides,
  };
  render(<MentionDropdown {...props} />);
  return props;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("MentionDropdown", () => {
  it("shows an error message when loading failed", () => {
    renderDropdown({ isError: true });
    expect(screen.getByText(/couldn't load files/i)).toBeTruthy();
  });

  it("shows a loading message while searching", () => {
    renderDropdown({ isLoading: true });
    expect(screen.getByText(/searching files/i)).toBeTruthy();
  });

  it("shows an empty state when there are no matches", () => {
    renderDropdown();
    expect(screen.getByText(/no matching files/i)).toBeTruthy();
  });

  it("renders a row per file and selects on mousedown", () => {
    const { onSelect } = renderDropdown({ options: [fileOption(FILE)] });
    const option = screen.getByRole("option", { name: /alpha\.txt/i });
    fireEvent.mouseDown(option);
    expect(onSelect).toHaveBeenCalledWith(fileOption(FILE));
  });

  it("highlights a row on hover", () => {
    const { onHighlight } = renderDropdown({
      options: [fileOption(FILE)],
      highlightedIndex: -1,
    });
    fireEvent.mouseEnter(screen.getByRole("option", { name: /alpha\.txt/i }));
    expect(onHighlight).toHaveBeenCalledWith(0);
  });

  it("offers folders above files, with their count in the accessible name", () => {
    renderDropdown({
      options: [folderOption(FOLDER, 1), fileOption(FILE)],
    });
    const options = screen.getAllByRole("option");
    expect(options[0].getAttribute("aria-label")).toBe(
      "Folder Reports, 2 files",
    );
    expect(options[1].textContent).toContain("alpha.txt");
  });

  it("selects a folder option on mousedown", () => {
    const { onSelect } = renderDropdown({
      options: [folderOption(FOLDER, 0)],
    });
    fireEvent.mouseDown(screen.getByRole("option", { name: /Folder Reports/ }));
    expect(onSelect).toHaveBeenCalledWith(folderOption(FOLDER, 0));
  });
});

const FOLDER: WorkspaceFolder = {
  id: "fld-1",
  workspace_id: "ws-1",
  name: "Reports",
  parent_id: null,
  file_count: 2,
  created_at: new Date("2026-01-01T00:00:00Z"),
  updated_at: new Date("2026-01-01T00:00:00Z"),
};

function fileOption(file: WorkspaceFileItem): MentionOption {
  return { kind: "file", file };
}

function folderOption(
  folder: WorkspaceFolder,
  subfolderCount: number,
): MentionOption {
  return { kind: "folder", folder, subfolderCount };
}
