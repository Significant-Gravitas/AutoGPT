import {
  act,
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState, type MutableRefObject } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { FullscreenDialog } from "@/components/molecules/FullscreenDialog/FullscreenDialog";
import {
  connectedIntegrationsFromCredentials,
  type IntegrationMention,
} from "../../helpers";
import { MentionDropdown } from "../MentionDropdown";
import { useChatMentions, type MentionOption } from "../../useChatMentions";

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
const GOOGLE: IntegrationMention = {
  credentialId: "google-1",
  providerName: "Google",
  username: null,
  provider: "google",
  name: "Google",
  token: "@Google",
};
const GOOGLE_ITEM: MentionOption = { kind: "integration", integration: GOOGLE };

type DropdownProps = Parameters<typeof MentionDropdown>[0];

function dropdownProps(overrides: Partial<DropdownProps> = {}): DropdownProps {
  return {
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
}

function renderDropdown(overrides: Partial<DropdownProps> = {}) {
  const props = dropdownProps(overrides);
  render(<MentionDropdown {...props} />);
  return props;
}

function ComposerWithDropdown({
  open,
  onSelect = vi.fn(),
}: {
  open: boolean;
  onSelect?: DropdownProps["onSelect"];
}) {
  return (
    <form className="relative">
      {open && (
        <MentionDropdown
          {...dropdownProps({ options: [FILE_ITEM], onSelect })}
        />
      )}
      <textarea aria-label="Chat message" />
    </form>
  );
}

function MentionComposer() {
  const [value, setValue] = useState("");
  const mentions = useChatMentions({
    enabled: true,
    value,
    setValue,
    includeWorkspaceFiles: false,
    integrations: [GOOGLE],
    addWorkspaceFile: vi.fn(),
    addWorkspaceFolder: vi.fn(),
  });
  return (
    <form className="relative">
      {mentions.isOpen && (
        <MentionDropdown
          {...mentions}
          onSelect={mentions.accept}
          onHighlight={mentions.setHighlightedIndex}
        />
      )}
      <textarea
        aria-label="Chat message"
        value={value}
        onChange={(e) => {
          setValue(e.currentTarget.value);
          mentions.detect(e.currentTarget);
        }}
        onKeyDown={mentions.onKeyDown}
        onBlur={mentions.close}
      />
    </form>
  );
}

function openMentionList() {
  const textbox = screen.getByRole("textbox", { name: "Chat message" });
  act(() => textbox.focus());
  fireEvent.change(textbox, { target: { value: "@", selectionStart: 1 } });
  expect(
    screen.getByRole("listbox", { name: "Mention suggestions" }),
  ).toBeTruthy();
  return textbox as HTMLTextAreaElement;
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

  it("uses the fixed positioning strategy inside a scrolling ancestor", () => {
    render(
      <div className="overflow-y-auto">
        <form className="relative">
          <MentionDropdown {...dropdownProps({ options: [FILE_ITEM] })} />
        </form>
      </div>,
    );
    const list = screen.getByRole("listbox", { name: "Mention suggestions" });
    expect(list.style.position).toBe("fixed");
  });

  it("keeps focus in the chat textbox when the list opens", () => {
    const { rerender } = render(<ComposerWithDropdown open={false} />);
    const textbox = screen.getByRole("textbox");
    textbox.focus();
    rerender(<ComposerWithDropdown open />);
    expect(document.activeElement).toBe(textbox);
    const list = screen.getByRole("listbox", { name: "Mention suggestions" });
    expect(
      within(list).getByRole("option", { name: /alpha\.txt/ }),
    ).toBeTruthy();
  });

  it("selects an option on click without moving focus", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const { rerender } = render(
      <ComposerWithDropdown open={false} onSelect={onSelect} />,
    );
    const textbox = screen.getByRole("textbox");
    textbox.focus();
    rerender(<ComposerWithDropdown open onSelect={onSelect} />);
    await user.click(screen.getByRole("option", { name: /alpha\.txt/ }));
    expect(onSelect).toHaveBeenCalledWith(FILE_ITEM);
    expect(document.activeElement).toBe(textbox);
  });

  it("selects a folder option on mousedown", () => {
    const { onSelect } = renderDropdown({
      options: [folderOption(FOLDER, 0)],
    });
    fireEvent.mouseDown(screen.getByRole("option", { name: /Folder Reports/ }));
    expect(onSelect).toHaveBeenCalledWith(folderOption(FOLDER, 0));
  });
});

describe("MentionDropdown in the composer", () => {
  it.each([{ isComposing: true }, { keyCode: 229 }])(
    "leaves a composing Escape to the input method (%j)",
    (composing) => {
      render(<MentionComposer />);
      const textbox = openMentionList();
      const escape = new KeyboardEvent("keydown", {
        key: "Escape",
        bubbles: true,
        cancelable: true,
        ...composing,
      });
      fireEvent(textbox, escape);
      expect(escape.defaultPrevented).toBe(false);
      expect(
        screen.getByRole("listbox", { name: "Mention suggestions" }),
      ).toBeTruthy();
    },
  );

  it("closes on Escape and keeps focus in the chat textbox", () => {
    render(<MentionComposer />);
    const textbox = openMentionList();
    const escape = new KeyboardEvent("keydown", {
      key: "Escape",
      bubbles: true,
      cancelable: true,
    });
    fireEvent(textbox, escape);
    expect(escape.defaultPrevented).toBe(true);
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(document.activeElement).toBe(textbox);
  });

  it("selects an option inside a dialog without closing the dialog", () => {
    const onClose = vi.fn();
    render(
      <FullscreenDialog title="Expert chat" onClose={onClose}>
        <MentionComposer />
      </FullscreenDialog>,
    );
    const textbox = openMentionList();
    const option = screen.getByRole("option", { name: /Google/ });
    fireEvent.pointerDown(option, { pointerType: "mouse" });
    fireEvent.mouseDown(option);
    expect(textbox.value).toBe("@Google ");
    expect(screen.queryByRole("listbox")).toBeNull();
    expect(screen.getByRole("dialog", { name: "Expert chat" })).toBeTruthy();
    expect(onClose).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(textbox);
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
