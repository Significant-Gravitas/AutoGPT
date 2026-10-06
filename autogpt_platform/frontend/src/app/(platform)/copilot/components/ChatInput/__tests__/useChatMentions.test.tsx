import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor,
} from "@testing-library/react";
import React, { useState, type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { connectedIntegrationsFromCredentials } from "../helpers";
import { useChatMentions } from "../useChatMentions";

const mockListWorkspaceFiles = vi.fn();
const mockFolders = vi.fn(() => ({ data: { folders: [] } }));
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

function Wrapper({ children }: { children: ReactNode }) {
  const [client] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false } },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function fakeTextarea(value: string, caret = value.length) {
  return { value, selectionStart: caret } as HTMLTextAreaElement;
}

function keyEvent(key: string, composing: boolean | "keyCode229" = false) {
  const isComposing = composing === true;
  // Safari confirms a candidate with an Enter fired after compositionend, so
  // isComposing is already false and only the legacy keyCode is left.
  const keyCode = composing === "keyCode229" ? 229 : key === "Enter" ? 13 : 0;
  return {
    key,
    nativeEvent: { key, isComposing, keyCode },
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLTextAreaElement>;
}

const GOOGLE = {
  credentialId: "google-1",
  providerName: "Google",
  username: null,
  provider: "google",
  name: "Google",
  token: "@Google",
};
const GITHUB = {
  credentialId: "github-1",
  providerName: "GitHub",
  username: null,
  provider: "github",
  name: "GitHub",
  token: "@GitHub",
};

afterEach(() => {
  vi.clearAllMocks();
});

describe("useChatMentions", () => {
  it("opens on an @token at the caret and queries workspace files", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @al",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    expect(result.current.isOpen).toBe(true);

    await waitFor(() => expect(result.current.options).toHaveLength(1));
    await waitFor(() =>
      expect(mockListWorkspaceFiles).toHaveBeenCalledWith({
        limit: 8,
        q: "al",
      }),
    );
  });

  it("stays closed when there is no @token before the caret", () => {
    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hello world",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hello world")));
    expect(result.current.isOpen).toBe(false);
  });

  it("does not open while disabled", () => {
    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: false,
          value: "hi @al",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    expect(result.current.isOpen).toBe(false);
  });

  it("strips the @query and attaches the file when accepted via Enter", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const setValue = vi.fn();
    const addWorkspaceFile = vi.fn();

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @al",
          setValue,
          addWorkspaceFile,
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.options).toHaveLength(1));

    let handled = false;
    act(() => {
      handled = result.current.onKeyDown(keyEvent("Enter"));
    });

    expect(handled).toBe(true);
    expect(setValue).toHaveBeenCalledWith("hi ");
    expect(addWorkspaceFile).toHaveBeenCalledWith(FILE);
    expect(result.current.isOpen).toBe(false);
  });

  it("closes on Escape without attaching", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const addWorkspaceFile = vi.fn();

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @al",
          setValue: vi.fn(),
          addWorkspaceFile,
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.options).toHaveLength(1));

    act(() => {
      result.current.onKeyDown(keyEvent("Escape"));
    });

    expect(result.current.isOpen).toBe(false);
    expect(addWorkspaceFile).not.toHaveBeenCalled();
  });

  it("navigates results with arrow keys but ignores composing keydowns", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: {
        files: [FILE, { ...FILE, id: "file-2", name: "beta.txt" }],
        has_more: false,
      },
    });

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @")));
    await waitFor(() => expect(result.current.options).toHaveLength(2));

    act(() => {
      expect(result.current.onKeyDown(keyEvent("ArrowDown"))).toBe(true);
    });
    expect(result.current.highlightedIndex).toBe(1);

    act(() => {
      expect(result.current.onKeyDown(keyEvent("ArrowUp"))).toBe(true);
    });
    expect(result.current.highlightedIndex).toBe(0);

    act(() => {
      expect(result.current.onKeyDown(keyEvent("ArrowDown", true))).toBe(false);
    });
    expect(result.current.highlightedIndex).toBe(0);
  });

  it("does not accept a mention on a composing Enter or Tab", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const setValue = vi.fn();
    const addWorkspaceFile = vi.fn();

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @al",
          setValue,
          addWorkspaceFile,
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.options).toHaveLength(1));

    for (const event of [
      keyEvent("Enter", true),
      keyEvent("Enter", "keyCode229"),
      keyEvent("Tab", true),
    ]) {
      act(() => {
        expect(result.current.onKeyDown(event)).toBe(false);
      });
      expect(event.preventDefault).not.toHaveBeenCalled();
    }

    expect(setValue).not.toHaveBeenCalled();
    expect(addWorkspaceFile).not.toHaveBeenCalled();
    expect(result.current.isOpen).toBe(true);
  });

  it("ignores accept when the highlighted item is out of bounds", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });
    const setValue = vi.fn();
    const addWorkspaceFile = vi.fn();

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @al",
          setValue,
          addWorkspaceFile,
          addWorkspaceFolder: vi.fn(),
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.options).toHaveLength(1));

    // A shrinking result list can leave the highlighted index pointing past
    // the end before the clamp effect runs — accepting that must be a no-op,
    // not a crash on an undefined item.
    act(() => result.current.accept(undefined));

    expect(setValue).not.toHaveBeenCalled();
    expect(addWorkspaceFile).not.toHaveBeenCalled();
  });

  it("lists matching integrations above the file results", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [FILE], has_more: false },
    });

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @g",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
          integrations: [GITHUB, GOOGLE],
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @g")));
    expect(result.current.hasIntegrations).toBe(true);
    expect(result.current.options).toEqual([
      { kind: "integration", integration: GITHUB },
      { kind: "integration", integration: GOOGLE },
    ]);

    await waitFor(() => expect(result.current.options).toHaveLength(3));
    expect(result.current.options[2]).toEqual({ kind: "file", file: FILE });
  });

  it("narrows integrations by the typed query without waiting on the debounce", () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [], has_more: false },
    });

    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @goo",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
          integrations: [GITHUB, GOOGLE],
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @goo")));
    expect(result.current.options).toEqual([
      { kind: "integration", integration: GOOGLE },
    ]);
  });

  it("inserts the integration token into the prompt and moves the caret after it", async () => {
    mockListWorkspaceFiles.mockResolvedValue({
      status: 200,
      data: { files: [], has_more: false },
    });
    const addWorkspaceFile = vi.fn();

    function Harness() {
      const [value, setValue] = useState("check ");
      const mentions = useChatMentions({
        enabled: true,
        value,
        setValue,
        addWorkspaceFile,
        addWorkspaceFolder: vi.fn(),
        integrations: [GOOGLE],
      });
      return (
        <textarea
          aria-label="composer"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            mentions.detect(e.currentTarget);
          }}
          onKeyDown={(e) => mentions.onKeyDown(e)}
        />
      );
    }

    render(<Harness />, { wrapper: Wrapper });
    const textarea = screen.getByLabelText<HTMLTextAreaElement>("composer");

    // Caret sits right after "@goo", with more text following it.
    fireEvent.change(textarea, {
      target: {
        value: "check @goo for me",
        selectionStart: "check @goo".length,
        selectionEnd: "check @goo".length,
      },
    });
    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(textarea.value).toBe("check @Google for me");
    expect(addWorkspaceFile).not.toHaveBeenCalled();
    // The browser would leave the caret at the end after the value swap; the
    // hook parks it right after the inserted mention instead.
    await waitFor(() =>
      expect(textarea.selectionStart).toBe("check @Google ".length),
    );
  });

  it("accepts two named accounts from one provider, including a query with spaces", () => {
    const integrations = connectedIntegrationsFromCredentials([
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
    function Harness() {
      const [value, setValue] = useState("");
      const mentions = useChatMentions({
        enabled: true,
        value,
        setValue,
        integrations,
        includeWorkspaceFiles: false,
        addWorkspaceFile: vi.fn(),
        addWorkspaceFolder: vi.fn(),
      });
      return (
        <textarea
          aria-label="accounts"
          value={value}
          onChange={(e) => {
            setValue(e.target.value);
            mentions.detect(e.currentTarget);
          }}
          onKeyDown={(e) => mentions.onKeyDown(e)}
        />
      );
    }
    render(<Harness />, { wrapper: Wrapper });
    const input = screen.getByLabelText<HTMLTextAreaElement>("accounts");
    fireEvent.change(input, { target: { value: "Check my @Work G" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(input.value).toBe(
      "Check my [Work Gmail](credential://google/work) ",
    );
    fireEvent.change(input, {
      target: { value: `${input.value}for new TODOs, and @Personal` },
    });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(input.value).toBe(
      "Check my [Work Gmail](credential://google/work) for new TODOs, and [Personal Gmail](credential://google/personal) ",
    );
  });

  it("never opens when workspace files are off and nothing is connected", () => {
    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
          includeWorkspaceFiles: false,
          integrations: [],
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @")));
    expect(result.current.isOpen).toBe(false);
    expect(result.current.options).toEqual([]);
    expect(mockListWorkspaceFiles).not.toHaveBeenCalled();
  });

  it("offers only integrations and never queries files when workspace files are off", () => {
    const { result } = renderHook(
      () =>
        useChatMentions({
          enabled: true,
          value: "hi @",
          setValue: vi.fn(),
          addWorkspaceFile: vi.fn(),
          addWorkspaceFolder: vi.fn(),
          includeWorkspaceFiles: false,
          integrations: [GOOGLE],
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @")));
    expect(result.current.isOpen).toBe(true);
    expect(result.current.showFiles).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.options).toEqual([
      { kind: "integration", integration: GOOGLE },
    ]);
    expect(mockListWorkspaceFiles).not.toHaveBeenCalled();
  });
});
