import { getGetV2ListChatConnectionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useChatInput } from "../components/ChatInput/useChatInput";
import { useConnectionPicker } from "../components/ChatInput/components/ConnectionPicker/useConnectionPicker";
import { useLockedTierFallback } from "../components/ChatInput/components/ConnectionPicker/useLockedTierFallback";
import { DEFAULT_PANEL_WIDTH, useCopilotUIStore } from "../store";
import { useSessionDeletion } from "../useSessionDeletion";
import { useWorkflowImportAutoSubmit } from "../useWorkflowImportAutoSubmit";

// Each copilot UI store reader subscribes to the fields it reads, not the
// whole store: a write to a field it does not read (a panel resize writes
// `contextPanelWidth` every frame of a drag) leaves it alone, while a write
// to a field it does read still reaches it.

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-1" }, isLoggedIn: true }),
}));

function writeUnrelatedField() {
  act(() => {
    useCopilotUIStore
      .getState()
      .setContextPanelWidth(useCopilotUIStore.getState().contextPanelWidth + 7);
  });
}

async function settle() {
  await act(async () => {});
}

function HookProviders({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false } },
      }),
  );
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter>{children}</NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

function countHookRenders<T>(useHook: () => T) {
  const counter = { renders: 0 };
  const view = renderHook(
    () => {
      counter.renders++;
      return useHook();
    },
    { wrapper: HookProviders },
  );
  return { ...view, counter };
}

const initialState = useCopilotUIStore.getState();

beforeEach(() => {
  server.use(getGetV2ListChatConnectionsMockHandler200({ offers: [] }));
  useCopilotUIStore.setState({
    contextPanelWidth: DEFAULT_PANEL_WIDTH,
    copilotLlmModel: "standard",
    initialPrompt: null,
    sessionToDelete: null,
    isNotificationsEnabled: false,
    isSoundEnabled: true,
  });
});

afterEach(() => {
  useCopilotUIStore.setState(initialState);
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("copilot UI store readers", () => {
  it("useWorkflowImportAutoSubmit ignores unrelated writes", async () => {
    const { counter } = countHookRenders(() =>
      useWorkflowImportAutoSubmit({
        onSend: vi.fn().mockResolvedValue(undefined),
        setPendingFileParts: vi.fn(),
      }),
    );
    await settle();
    const before = counter.renders;

    writeUnrelatedField();

    expect(counter.renders).toBe(before);
  });

  it("useSessionDeletion ignores unrelated writes and follows sessionToDelete", async () => {
    const { result, counter } = countHookRenders(() => useSessionDeletion());
    await settle();
    const before = counter.renders;

    writeUnrelatedField();
    expect(counter.renders).toBe(before);

    act(() => {
      useCopilotUIStore
        .getState()
        .setSessionToDelete({ id: "s1", title: "A chat" });
    });
    expect(result.current.sessionToDelete).toEqual({
      id: "s1",
      title: "A chat",
    });
  });

  it("useLockedTierFallback ignores unrelated writes", async () => {
    const { counter } = countHookRenders(() =>
      useLockedTierFallback(undefined),
    );
    await settle();
    const before = counter.renders;

    writeUnrelatedField();
    expect(counter.renders).toBe(before);

    act(() => useCopilotUIStore.getState().setCopilotLlmModel("advanced"));
    expect(counter.renders).toBeGreaterThan(before);
  });

  it("useConnectionPicker ignores unrelated writes and follows the model", async () => {
    const { result, counter } = countHookRenders(() => useConnectionPicker());
    await waitFor(() => expect(result.current).toBeDefined());
    await settle();
    const before = counter.renders;

    writeUnrelatedField();
    expect(counter.renders).toBe(before);

    act(() => useCopilotUIStore.getState().setCopilotLlmModel("advanced"));
    expect(counter.renders).toBeGreaterThan(before);
  });

  it("useChatInput ignores unrelated writes and still picks up an initial prompt", async () => {
    const { result, counter } = countHookRenders(() =>
      useChatInput({ onSend: vi.fn(), inputId: "store-selectors-input" }),
    );
    await settle();
    const before = counter.renders;

    writeUnrelatedField();
    expect(counter.renders).toBe(before);

    act(() => useCopilotUIStore.getState().setInitialPrompt("Hello Otto"));
    await waitFor(() => expect(result.current.value).toBe("Hello Otto"));
  });
});
