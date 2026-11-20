import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, render, renderHook, screen } from "@testing-library/react";
import React, { type ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MentionDropdown } from "../components/MentionDropdown";
import {
  filterSkillCommands,
  insertSkillCommand,
  type SkillCommand,
} from "../helpers";
import { useChatMentions, type MentionOption } from "../useChatMentions";

const mockListWorkspaceFiles = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  listWorkspaceFiles: (...args: unknown[]) => mockListWorkspaceFiles(...args),
  useListWorkspaceFolders: () => ({ data: { folders: [] } }),
}));

const mockSkillCommands = vi.fn();
vi.mock("../useSkillCommands", () => ({
  useSkillCommands: (...args: unknown[]) => mockSkillCommands(...args),
}));

const FIX_ISSUE: SkillCommand = {
  name: "fix-issue",
  description: "Fix a GitHub issue by number",
  argumentHint: "[issue-number]",
};
const TRIAGE: SkillCommand = {
  name: "incident-triage",
  description: "Triage an incident and fix the paging",
  argumentHint: null,
};

function Wrapper({ children }: { children: ReactNode }) {
  const [client] = React.useState(
    () => new QueryClient({ defaultOptions: { queries: { retry: false } } }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function fakeTextarea(value: string, caret = value.length) {
  return {
    value,
    selectionStart: caret,
    setSelectionRange: vi.fn(),
  };
}

function enter() {
  return {
    key: "Enter",
    nativeEvent: { key: "Enter", isComposing: false, keyCode: 13 },
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent<HTMLTextAreaElement>;
}

function renderMentions(value: string, setValue = vi.fn()) {
  return renderHook(
    () =>
      useChatMentions({
        enabled: true,
        value,
        setValue,
        addWorkspaceFile: vi.fn(),
        addWorkspaceFolder: vi.fn(),
      }),
    { wrapper: Wrapper },
  );
}

function skillNames(options: MentionOption[]) {
  return options.map((option) =>
    option.kind === "skill" ? option.skill.name : option.kind,
  );
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("skill command helpers", () => {
  it("puts names that start with the query ahead of other matches", () => {
    expect(filterSkillCommands([TRIAGE, FIX_ISSUE], "fix")).toEqual([
      FIX_ISSUE,
      TRIAGE,
    ]);
    expect(filterSkillCommands([TRIAGE, FIX_ISSUE], "")).toEqual([
      TRIAGE,
      FIX_ISSUE,
    ]);
  });

  it("writes the command and a space so the arguments come next", () => {
    expect(insertSkillCommand("/fi", { start: 0, end: 3 }, FIX_ISSUE)).toEqual({
      value: "/fix-issue ",
      caret: 11,
    });
    expect(
      insertSkillCommand("/fi   42", { start: 0, end: 3 }, FIX_ISSUE),
    ).toEqual({ value: "/fix-issue 42", caret: 11 });
  });
});

describe("useChatMentions with a leading /", () => {
  it("lists the chat's skills without searching files", () => {
    mockSkillCommands.mockReturnValue({
      skills: [TRIAGE, FIX_ISSUE],
      isLoading: false,
    });
    const { result } = renderMentions("/fi");

    act(() => result.current.detect(fakeTextarea("/fi")));

    expect(result.current.isOpen).toBe(true);
    expect(result.current.trigger).toBe("/");
    expect(skillNames(result.current.options)).toEqual([
      "fix-issue",
      "incident-triage",
    ]);
    expect(mockListWorkspaceFiles).not.toHaveBeenCalled();
  });

  it("only asks for skills once a command is typed", () => {
    mockSkillCommands.mockReturnValue({
      skills: [FIX_ISSUE],
      isLoading: false,
    });
    const { result } = renderMentions("/");

    expect(mockSkillCommands).toHaveBeenLastCalledWith(undefined, false);
    act(() => result.current.detect(fakeTextarea("/")));
    expect(mockSkillCommands).toHaveBeenLastCalledWith(undefined, true);
  });

  it("writes the picked command into the message on Enter", () => {
    mockSkillCommands.mockReturnValue({
      skills: [FIX_ISSUE],
      isLoading: false,
    });
    const setValue = vi.fn();
    const { result } = renderMentions("/fi", setValue);

    act(() => result.current.detect(fakeTextarea("/fi")));
    act(() => {
      result.current.onKeyDown(enter());
    });

    expect(setValue).toHaveBeenCalledWith("/fix-issue ");
    expect(result.current.isOpen).toBe(false);
  });

  it.each([
    ["a / later in the message", "run /fix"],
    ["a path", "/etc/hosts"],
    ["a command with its arguments typed", "/fix-issue 42"],
  ])("stays closed for %s", (_, text) => {
    mockSkillCommands.mockReturnValue({
      skills: [FIX_ISSUE],
      isLoading: false,
    });
    const { result } = renderMentions(text);

    act(() => result.current.detect(fakeTextarea(text)));

    expect(result.current.isOpen).toBe(false);
  });

  it("stays closed when the chat has no skills to run", () => {
    mockSkillCommands.mockReturnValue({ skills: [], isLoading: false });
    const { result } = renderMentions("/");

    act(() => result.current.detect(fakeTextarea("/")));

    expect(result.current.isOpen).toBe(false);
  });

  it("opens in a loading state while the skills load", () => {
    mockSkillCommands.mockReturnValue({ skills: [], isLoading: true });
    const { result } = renderMentions("/");

    act(() => result.current.detect(fakeTextarea("/")));

    expect(result.current.isOpen).toBe(true);
    expect(result.current.isLoading).toBe(true);
  });
});

describe("MentionDropdown for skill commands", () => {
  function renderDropdown(options: MentionOption[], isLoading = false) {
    return render(
      <MentionDropdown
        trigger="/"
        options={options}
        showFiles={false}
        hasIntegrations={false}
        isLoading={isLoading}
        isError={false}
        highlightedIndex={0}
        highlightedRef={{ current: null }}
        onSelect={vi.fn()}
        onHighlight={vi.fn()}
      />,
    );
  }

  it("shows each command with its argument hint and description", () => {
    renderDropdown([{ kind: "skill", skill: FIX_ISSUE }]);

    expect(
      screen.getByRole("listbox", { name: "Skill commands" }),
    ).toBeDefined();
    expect(screen.getByText("/fix-issue")).toBeDefined();
    expect(screen.getByText("[issue-number]")).toBeDefined();
    expect(screen.getByText("Fix a GitHub issue by number")).toBeDefined();
  });

  it("says when nothing matches and while skills load", () => {
    const { rerender } = renderDropdown([]);
    expect(screen.getByText("No matching skills.")).toBeDefined();

    rerender(
      <MentionDropdown
        trigger="/"
        options={[]}
        showFiles={false}
        hasIntegrations={false}
        isLoading
        isError={false}
        highlightedIndex={0}
        highlightedRef={{ current: null }}
        onSelect={vi.fn()}
        onHighlight={vi.fn()}
      />,
    );
    expect(screen.getByText("Loading skills…")).toBeDefined();
  });
});
