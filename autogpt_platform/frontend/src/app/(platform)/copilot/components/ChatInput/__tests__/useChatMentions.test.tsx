import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import React, { type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useChatMentions } from "../useChatMentions";

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

function keyEvent(key: string) {
  return {
    key,
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLTextAreaElement>;
}

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
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    expect(result.current.isOpen).toBe(true);

    await waitFor(() => expect(result.current.files).toHaveLength(1));
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
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.files).toHaveLength(1));

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
        }),
      { wrapper: Wrapper },
    );

    act(() => result.current.detect(fakeTextarea("hi @al")));
    await waitFor(() => expect(result.current.files).toHaveLength(1));

    act(() => {
      result.current.onKeyDown(keyEvent("Escape"));
    });

    expect(result.current.isOpen).toBe(false);
    expect(addWorkspaceFile).not.toHaveBeenCalled();
  });
});
