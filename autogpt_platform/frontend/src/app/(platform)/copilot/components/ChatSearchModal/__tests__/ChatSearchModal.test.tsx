import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotChatRuntimeStore } from "../../../copilotChatRegistry";
import { useCopilotUIStore } from "../../../store";
import { ChatSearchModal } from "../ChatSearchModal";
import type { SearchSession } from "../helpers";

function makeSession(
  overrides: Partial<SearchSession> & { id: string },
): SearchSession {
  return {
    title: `Session ${overrides.id}`,
    updated_at: "2025-01-01T00:00:00Z",
    ...overrides,
  };
}

function renderModal(props: {
  sessions?: SearchSession[];
  currentSessionId?: string | null;
  onSelectSession?: (id: string) => void;
}) {
  const onSelectSession = props.onSelectSession ?? vi.fn();
  return {
    onSelectSession,
    ...render(
      <ChatSearchModal
        sessions={props.sessions ?? []}
        currentSessionId={props.currentSessionId ?? null}
        onSelectSession={onSelectSession}
      />,
    ),
  };
}

describe("ChatSearchModal", () => {
  beforeEach(() => {
    useCopilotUIStore.setState({
      isSearchOpen: true,
      completedSessionIDs: new Set<string>(),
    });
    useCopilotChatRuntimeStore.setState({ sessionNeedsReload: {} });
  });

  afterEach(() => {
    useCopilotUIStore.setState({ isSearchOpen: false });
  });

  it("returns null when the search modal is closed", () => {
    useCopilotUIStore.setState({ isSearchOpen: false });
    renderModal({ sessions: [makeSession({ id: "a" })] });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("auto-focuses the input when opened", async () => {
    renderModal({ sessions: [makeSession({ id: "a", title: "Alpha" })] });
    const input = await screen.findByRole("textbox", { name: /search chats/i });
    await vi.waitFor(() => expect(document.activeElement).toBe(input));
  });

  it("shows 'No chats found' when query has no matches and hides the clear button when empty", async () => {
    const user = userEvent.setup();
    renderModal({ sessions: [makeSession({ id: "a", title: "Alpha" })] });

    expect(screen.queryByRole("button", { name: /clear search/i })).toBeNull();

    const input = screen.getByRole("textbox", { name: /search chats/i });
    await user.type(input, "zzz");
    expect(await screen.findByText("No chats found")).toBeDefined();
    expect(screen.getByRole("button", { name: /clear search/i })).toBeDefined();
  });

  it("closes when the backdrop is clicked but not when the dialog body is clicked", async () => {
    renderModal({ sessions: [makeSession({ id: "a" })] });
    const dialog = await screen.findByRole("dialog");

    fireEvent.mouseDown(dialog);
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(true);

    const backdrop = dialog.parentElement!;
    fireEvent.mouseDown(backdrop);
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
  });

  it("closes when the close button is clicked", async () => {
    renderModal({ sessions: [makeSession({ id: "a" })] });
    fireEvent.click(
      await screen.findByRole("button", { name: /close search/i }),
    );
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
  });

  it("closes on Escape", async () => {
    renderModal({ sessions: [makeSession({ id: "a" })] });
    const dialog = await screen.findByRole("dialog");
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
  });

  it("navigates with ArrowDown/ArrowUp, clamps at the bounds, and selects via Enter", async () => {
    const onSelect = vi.fn();
    const sessions = [
      makeSession({
        id: "a",
        title: "Alpha",
        updated_at: "2025-01-03T00:00:00Z",
      }),
      makeSession({
        id: "b",
        title: "Beta",
        updated_at: "2025-01-02T00:00:00Z",
      }),
      makeSession({
        id: "c",
        title: "Gamma",
        updated_at: "2025-01-01T00:00:00Z",
      }),
    ];
    renderModal({ sessions, onSelectSession: onSelect });

    const dialog = await screen.findByRole("dialog");
    const options = within(dialog).getAllByRole("option");
    expect(options[0].getAttribute("aria-selected")).toBe("true");

    fireEvent.keyDown(dialog, { key: "ArrowUp" });
    expect(
      within(dialog).getAllByRole("option")[0].getAttribute("aria-selected"),
    ).toBe("true");

    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    expect(
      within(dialog).getAllByRole("option")[1].getAttribute("aria-selected"),
    ).toBe("true");

    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    fireEvent.keyDown(dialog, { key: "ArrowDown" });
    expect(
      within(dialog).getAllByRole("option")[2].getAttribute("aria-selected"),
    ).toBe("true");

    fireEvent.keyDown(dialog, { key: "Enter" });
    expect(onSelect).toHaveBeenCalledWith("c");
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
  });

  it("Enter is a no-op when there are no results", async () => {
    const onSelect = vi.fn();
    renderModal({
      sessions: [makeSession({ id: "a", title: "Alpha" })],
      onSelectSession: onSelect,
    });
    const user = userEvent.setup();
    const dialog = await screen.findByRole("dialog");
    await user.type(
      screen.getByRole("textbox", { name: /search chats/i }),
      "nope",
    );
    await screen.findByText("No chats found");
    fireEvent.keyDown(dialog, { key: "Enter" });
    expect(onSelect).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(true);
  });

  it("highlights a session on hover and selects it on click", async () => {
    const onSelect = vi.fn();
    renderModal({
      sessions: [
        makeSession({
          id: "a",
          title: "Alpha",
          updated_at: "2025-01-02T00:00:00Z",
        }),
        makeSession({
          id: "b",
          title: "Beta",
          updated_at: "2025-01-01T00:00:00Z",
        }),
      ],
      onSelectSession: onSelect,
    });
    const dialog = await screen.findByRole("dialog");
    const betaOption = within(dialog).getByRole("option", { name: /beta/i });
    fireEvent.mouseEnter(betaOption);
    expect(betaOption.getAttribute("aria-selected")).toBe("true");

    fireEvent.click(betaOption);
    expect(onSelect).toHaveBeenCalledWith("b");
    expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
  });

  it("clears the query when the clear button is clicked", async () => {
    const user = userEvent.setup();
    renderModal({ sessions: [makeSession({ id: "a", title: "Alpha" })] });
    const input = screen.getByRole("textbox", { name: /search chats/i });
    await user.type(input, "alp");
    expect(await screen.findByText("Results")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /clear search/i }));
    expect((input as HTMLInputElement).value).toBe("");
    expect(await screen.findByText("Recent chats")).toBeDefined();
  });

  it("renders running, queued, processing, and completed indicators", async () => {
    useCopilotUIStore.setState({
      isSearchOpen: true,
      completedSessionIDs: new Set<string>(["done"]),
    });
    useCopilotChatRuntimeStore.setState({
      sessionNeedsReload: { processing: false },
    });

    renderModal({
      currentSessionId: "current",
      sessions: [
        makeSession({
          id: "running",
          title: "Running chat",
          chat_status: "running",
          updated_at: "2025-01-05T00:00:00Z",
        }),
        makeSession({
          id: "queued",
          title: "Queued chat",
          chat_status: "queued",
          updated_at: "2025-01-04T00:00:00Z",
        }),
        makeSession({
          id: "processing",
          title: "Processing chat",
          is_processing: true,
          updated_at: "2025-01-03T00:00:00Z",
        }),
        makeSession({
          id: "done",
          title: "Done chat",
          updated_at: "2025-01-02T00:00:00Z",
        }),
        makeSession({
          id: "current",
          title: "Current chat",
          updated_at: "2025-01-01T00:00:00Z",
        }),
      ],
    });

    expect(await screen.findByLabelText("Session running")).toBeDefined();
    expect(screen.getByLabelText("Session queued")).toBeDefined();
    expect(screen.getByLabelText("Session processing")).toBeDefined();
    expect(screen.getByLabelText("Session completed")).toBeDefined();
  });

  it("hides the processing indicator when needsReload is set for that session", async () => {
    useCopilotChatRuntimeStore.setState({
      sessionNeedsReload: { stale: true },
    });
    renderModal({
      sessions: [
        makeSession({
          id: "stale",
          title: "Stale chat",
          is_processing: true,
        }),
      ],
    });
    await screen.findByText("Stale chat");
    expect(screen.queryByLabelText("Session processing")).toBeNull();
  });

  it("falls back to 'Untitled chat' when the session has no title", async () => {
    renderModal({ sessions: [makeSession({ id: "a", title: null })] });
    expect(await screen.findByText("Untitled chat")).toBeDefined();
  });

  it("debounces query changes and bolds the matched substring", async () => {
    const user = userEvent.setup();
    renderModal({
      sessions: [makeSession({ id: "a", title: "Revenue forecast" })],
    });
    const input = screen.getByRole("textbox", { name: /search chats/i });
    await user.type(input, "fore");

    const option = await screen.findByRole("option", {
      name: /revenue forecast/i,
    });
    const bold = await within(option).findByText("fore");
    expect(bold.className).toContain("font-semibold");
  });
});
