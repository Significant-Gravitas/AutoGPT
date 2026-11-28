import { act } from "@testing-library/react";
import { hydrateRoot, type Root } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CopilotPage } from "../CopilotPage";

// What the chat under the frame reports, and what it was sent; the harness
// composer below uses the real hand-over hooks.
const chat = vi.hoisted(() => {
  interface ChatState {
    loaded: boolean;
    flagReady: boolean;
    disabled: boolean;
    draft: string;
    generation: number;
  }
  const listeners = new Set<() => void>();
  const holder = {
    state: {
      loaded: false,
      flagReady: true,
      disabled: false,
      draft: "",
      generation: 0,
    } as ChatState,
    sent: [] as string[],
    set(next: Partial<ChatState>) {
      holder.state = { ...holder.state, ...next };
      listeners.forEach((listener) => listener());
    },
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  };
  return holder;
});

vi.mock("../CopilotChatHost", async () => {
  const React = await import("react");
  const { useReportChatLoaded } = await import(
    "../components/ChatFrame/chatFrameHandoff"
  );
  const { useComposerHandoff } = await import(
    "../components/ChatInput/useComposerHandoff"
  );
  function useChatState() {
    return React.useSyncExternalStore(
      chat.subscribe,
      () => chat.state,
      () => chat.state,
    );
  }
  function Composer() {
    const { loaded, disabled } = useChatState();
    const [value, setValue] = React.useState(chat.state.draft);
    useReportChatLoaded(loaded);
    useComposerHandoff({
      inputId: "chat-input-empty",
      takesFrame: true,
      value,
      setValue(next) {
        chat.state.draft = next;
        setValue(next);
      },
      autoFocus: true,
    });
    return (
      <form
        onSubmit={(event) => {
          event.preventDefault();
          chat.sent.push(value);
        }}
      >
        <textarea
          id="chat-input-empty"
          aria-label="Chat message input"
          value={value}
          disabled={disabled}
          onChange={(event) => setValue(event.target.value)}
        />
      </form>
    );
  }
  function CopilotChatHost() {
    const { generation } = useChatState();
    return <Composer key={generation} />;
  }
  return { CopilotChatHost };
});
vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const React = await import("react");
  return {
    Flag: { COPILOT_STREAM_RUNTIME: "COPILOT_STREAM_RUNTIME" },
    useGetFlag: () => false,
    useFlagStatus() {
      const ready = React.useSyncExternalStore(
        chat.subscribe,
        () => chat.state.flagReady,
        () => false,
      );
      return { enabled: false, ready, answered: ready };
    },
  };
});
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isUserLoading: false, isLoggedIn: true }),
}));
vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => [null, vi.fn()],
}));
vi.mock("@/components/molecules/Sidebar/Sidebar", () => ({
  SidebarProvider: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));
vi.mock("../components/FileDropZone/FileDropZone", () => ({
  FileDropZone: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));
vi.mock(
  "@/components/layout/TopUpPrompt/LowCreditBanner/LowCreditBanner",
  () => ({ LowCreditBanner: () => null }),
);
vi.mock("../components/NotificationBanner/NotificationBanner", () => ({
  NotificationBanner: () => null,
}));
vi.mock("../components/ContextPanel/ContextPanelAutoOpen", () => ({
  ContextPanelAutoOpen: () => null,
}));
vi.mock("../components/CopilotModals/CopilotModals", () => ({
  CopilotModals: () => null,
}));
vi.mock("../useIsMobile", () => ({ useIsMobile: () => false }));

let container: HTMLDivElement;
let root: Root | null = null;

beforeEach(() => {
  chat.set({
    loaded: false,
    flagReady: true,
    disabled: false,
    draft: "",
    generation: 0,
  });
  chat.sent.length = 0;
  container = document.createElement("div");
  document.body.appendChild(container);
});

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  container.remove();
});

describe("the chat frame's message box", () => {
  it("is a real, enabled text box in the server's HTML", () => {
    container.innerHTML = renderToString(<CopilotPage />);

    const box = frameBox();
    expect(box?.disabled).toBe(false);
    expect(box?.closest("[inert]")).toBeNull();
    expect(chatBox()?.closest("[inert]")).not.toBeNull();
  });

  it("keeps text, caret and focus typed before React, through hydration and the hand-over", async () => {
    typeIntoServerFrame("hello world", 5);

    await hydrate();
    expect(frameBox()?.value).toBe("hello world");
    expect(document.activeElement).toBe(frameBox());

    act(() => chat.set({ loaded: true }));

    expect(frameBox()).toBeNull();
    const box = chatBox();
    expect(box?.value).toBe("hello world");
    expect([box?.selectionStart, box?.selectionEnd]).toEqual([5, 5]);
    expect(document.activeElement).toBe(box);
  });

  it("stays the message box until the chat-runtime flag has answered", async () => {
    typeIntoServerFrame("hello", 5);
    chat.set({ loaded: true, flagReady: false });

    await hydrate();
    expect(frameBox()?.value).toBe("hello");

    act(() => chat.set({ flagReady: true }));
    expect(frameBox()).toBeNull();
    expect(chatBox()?.value).toBe("hello");
  });

  it("sends a message submitted before React exactly once, when the chat takes the box", async () => {
    typeIntoServerFrame("plan my week", 12);
    runFrameScript();
    frameBox()?.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );
    expect(frameRoot()?.hasAttribute("data-send-pending")).toBe(true);

    await hydrate();
    expect(chat.sent).toEqual([]);

    act(() => chat.set({ loaded: true }));
    expect(chat.sent).toEqual(["plan my week"]);
    act(() => chat.set({ draft: chat.state.draft }));
    expect(chat.sent).toEqual(["plan my week"]);
  });

  it("puts what was typed after a draft the composer already had", async () => {
    chat.set({ draft: "Earlier draft" });
    typeIntoServerFrame("new words", 3);

    await hydrate();
    act(() => chat.set({ loaded: true }));

    const box = chatBox();
    expect(box?.value).toBe("Earlier draft\n\nnew words");
    expect(box?.selectionStart).toBe("Earlier draft\n\n".length + 3);
  });

  it("gives focus and a pending send to a composer that was disabled at the hand-over once it is enabled", async () => {
    chat.set({ disabled: true });
    typeIntoServerFrame("go", 2);
    runFrameScript();
    frameBox()?.dispatchEvent(
      new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
    );

    await hydrate();
    act(() => chat.set({ loaded: true }));
    expect(chat.sent).toEqual([]);

    act(() => chat.set({ disabled: false }));
    expect(document.activeElement).toBe(chatBox());
    expect(chat.sent).toEqual(["go"]);
  });

  it("keeps the caret and focus when the chat swaps its composer for a new one", async () => {
    typeIntoServerFrame("hello world", 5);
    await hydrate();
    act(() => chat.set({ loaded: true }));
    chatBox()?.setSelectionRange(2, 4);

    act(() => chat.set({ generation: 1 }));

    const box = chatBox();
    expect(box?.value).toBe("hello world");
    expect([box?.selectionStart, box?.selectionEnd]).toEqual([2, 4]);
    expect(document.activeElement).toBe(box);
  });
});

function frameRoot() {
  return container.querySelector<HTMLElement>('[data-testid="chat-frame"]');
}

function frameBox() {
  return container.querySelector<HTMLTextAreaElement>("#chat-frame-input");
}

function chatBox() {
  return container.querySelector<HTMLTextAreaElement>("#chat-input-empty");
}

// What a person does in the server's HTML before any script has loaded.
function typeIntoServerFrame(text: string, caret: number) {
  container.innerHTML = renderToString(<CopilotPage />);
  const box = frameBox();
  if (!box) throw new Error("the server's HTML has no frame box");
  box.value = text;
  box.focus();
  box.setSelectionRange(caret, caret);
}

// The test DOM runs no script, so run the server HTML's inline one as the
// browser does, with itself as `document.currentScript`.
function runFrameScript() {
  const script = frameRoot()?.querySelector("script");
  if (!script?.textContent) throw new Error("the frame has no inline script");
  const current = vi
    .spyOn(document, "currentScript", "get")
    .mockReturnValue(script);
  new Function(script.textContent)();
  current.mockRestore();
}

async function hydrate() {
  await act(async () => {
    root = hydrateRoot(container, <CopilotPage />, {
      onRecoverableError(error) {
        throw error;
      },
    });
  });
}
