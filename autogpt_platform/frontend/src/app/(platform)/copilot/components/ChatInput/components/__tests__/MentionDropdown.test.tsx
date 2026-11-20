import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { MutableRefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { connectedIntegrationsFromCredentials } from "../../helpers";
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
const FILE_ITEM: MentionOption = { kind: "file", file: FILE };
const GOOGLE_ITEM: MentionOption = {
  kind: "integration",
  integration: {
    credentialId: "google-1",
    providerName: "Google",
    username: null,
    provider: "google",
    name: "Google",
    token: "@Google",
  },
};

function renderDropdown(
  overrides: Partial<Parameters<typeof MentionDropdown>[0]> = {},
) {
  const props = {
    options: [] as MentionOption[],
    showFiles: true,
    hasIntegrations: false,
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
    expect(screen.getByText("No matching files.")).toBeTruthy();
  });

  it("names both kinds in the empty state when integrations are connected", () => {
    renderDropdown({ hasIntegrations: true });
    expect(screen.getByText("No matching files or integrations.")).toBeTruthy();
  });

  it("explains an integrations-only picker with nothing connected", () => {
    renderDropdown({ showFiles: false });
    expect(
      screen.getByText("No connected integrations to mention."),
    ).toBeTruthy();
  });

  it("renders a row per file and selects on mousedown", () => {
    const { onSelect } = renderDropdown({ options: [fileOption(FILE)] });
    const option = screen.getByRole("option", { name: /alpha\.txt/i });
    fireEvent.mouseDown(option);
    expect(onSelect).toHaveBeenCalledWith(fileOption(FILE));
    expect(screen.queryByText("Files")).toBeNull();
  });

  it("renders integrations with their token under a heading above files", () => {
    const { onSelect } = renderDropdown({
      options: [GOOGLE_ITEM, FILE_ITEM],
      hasIntegrations: true,
    });
    const options = screen.getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "GoogleGoogle",
      "alpha.txt",
    ]);
    expect(screen.getByText("Integrations")).toBeTruthy();
    expect(screen.getByText("Files")).toBeTruthy();

    fireEvent.mouseDown(options[0]);
    expect(onSelect).toHaveBeenCalledWith(GOOGLE_ITEM);
  });

  it("shows separate accounts with their names and usernames and selects the chosen credential", () => {
    const accounts = connectedIntegrationsFromCredentials([
      {
        id: "work",
        provider: "google",
        type: "oauth2",
        title: "Work Gmail",
        username: "work@example.com",
        scopes: null,
      },
      {
        id: "personal",
        provider: "google",
        type: "oauth2",
        title: "Personal Gmail",
        username: "me@example.com",
        scopes: null,
      },
    ]);
    const options: MentionOption[] = accounts.map((integration) => ({
      kind: "integration",
      integration,
    }));
    const { onSelect } = renderDropdown({ options, hasIntegrations: true });
    expect(screen.getAllByRole("option")).toHaveLength(2);
    const work = screen.getByRole("option", {
      name: /Work Gmail.*work@example.com/,
    });
    fireEvent.mouseDown(work);
    expect(onSelect).toHaveBeenCalledWith(options[1]);
    expect(
      screen.getByRole("option", { name: /Personal Gmail.*me@example.com/ }),
    ).toBeTruthy();
  });

  it("names an MCP credential by its service and server, never as MCP", () => {
    const [sentry] = connectedIntegrationsFromCredentials([
      {
        id: "cred-sentry",
        provider: "mcp",
        type: "oauth2",
        title: "MCP: mcp.sentry.dev",
        username: null,
        scopes: null,
        service: "sentry",
        service_name: "Sentry",
      },
    ]);
    renderDropdown({
      options: [{ kind: "integration", integration: sentry }],
      hasIntegrations: true,
    });
    const option = screen.getByRole("option");
    expect(option.textContent).toContain("mcp.sentry.dev");
    expect(option.textContent).toContain("Sentry");
    expect(option.textContent).not.toMatch(/mcp:|\bMcp\b|\bMCP\b/);
  });

  it("offers to connect an integration after the empty state, searching for what was typed", () => {
    const connect: MentionOption = { kind: "connect", query: "linear" };
    const { onSelect } = renderDropdown({
      options: [connect],
      hasIntegrations: true,
    });
    expect(screen.getByText("No matching files or integrations.")).toBeTruthy();
    const option = screen.getByRole("option", {
      name: /Connect an integration/,
    });
    expect(option.getAttribute("aria-selected")).toBe("true");
    expect(option.textContent).toContain("Search services for “linear”");
    fireEvent.mouseDown(option);
    expect(onSelect).toHaveBeenCalledWith(connect);
  });

  it("keeps connecting an integration as the last row below results", () => {
    const { onHighlight } = renderDropdown({
      options: [GOOGLE_ITEM, FILE_ITEM, { kind: "connect", query: "" }],
      hasIntegrations: true,
      highlightedIndex: 0,
    });
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(3);
    expect(options[2].textContent).toBe("Connect an integration");
    expect(screen.queryByText(/No matching/)).toBeNull();
    fireEvent.mouseEnter(options[2]);
    expect(onHighlight).toHaveBeenCalledWith(2);
  });

  it("still renders Connect last when the results are swapped for the next query", () => {
    const props = renderDropdown({
      options: [FILE_ITEM, { kind: "connect", query: "g" }],
      hasIntegrations: true,
    });
    const highlightedRef = props.highlightedRef;
    const next: MentionOption[] = [
      GOOGLE_ITEM,
      FILE_ITEM,
      { kind: "connect", query: "goo" },
    ];
    cleanup();
    render(
      <MentionDropdown
        {...props}
        highlightedRef={highlightedRef}
        options={next}
      />,
    );
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(3);
    expect(options[2].textContent).toBe(
      "Connect an integrationSearch services for “goo”",
    );
  });

  it("drops the headings when files are not part of the picker", () => {
    renderDropdown({
      options: [GOOGLE_ITEM],
      showFiles: false,
      hasIntegrations: true,
    });
    expect(screen.getByRole("option", { name: /google/i })).toBeTruthy();
    expect(screen.queryByText("Integrations")).toBeNull();
  });

  it("highlights a row on hover using its position in the combined list", () => {
    const { onHighlight } = renderDropdown({
      options: [GOOGLE_ITEM, FILE_ITEM],
      hasIntegrations: true,
      highlightedIndex: -1,
    });
    fireEvent.mouseEnter(screen.getByRole("option", { name: /alpha\.txt/i }));
    expect(onHighlight).toHaveBeenCalledWith(1);
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

  it("lists integrations, then folders, then files under the shared headings", () => {
    renderDropdown({
      options: [GOOGLE_ITEM, folderOption(FOLDER, 1), fileOption(FILE)],
      hasIntegrations: true,
    });
    const options = screen.getAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "GoogleGoogle",
      "Reports2 files",
      "alpha.txt",
    ]);
    expect(screen.getByText("Integrations")).toBeTruthy();
    expect(screen.getByText("Files")).toBeTruthy();
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
