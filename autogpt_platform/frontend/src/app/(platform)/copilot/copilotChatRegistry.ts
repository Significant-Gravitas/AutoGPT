import { Chat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { create } from "zustand";
import { useConnectedProvidersStore } from "./connectedProvidersStore";
import { useCopilotStreamStore } from "./copilotStreamStore";
import {
  createCopilotTransport,
  type MutableValue,
} from "./copilotStreamTransport";
import type { CopilotLlmModel, CopilotMode } from "./store";

interface CopilotChatRuntime {
  chat: Chat<UIMessage>;
  copilotModeRef: MutableValue<CopilotMode | undefined>;
  copilotModelRef: MutableValue<CopilotLlmModel | undefined>;
  onFinish?: (args: {
    isDisconnect?: boolean;
    isAbort?: boolean;
  }) => void | Promise<void>;
  onError?: (error: Error) => void;
}

interface CopilotChatRuntimeStore {
  sessionNeedsReload: Record<string, boolean>;
  markNeedsReload: (sessionId: string) => void;
  clearNeedsReload: (sessionId: string) => void;
  resetAll: () => void;
}

export const useCopilotChatRuntimeStore = create<CopilotChatRuntimeStore>(
  (set) => ({
    sessionNeedsReload: {},
    markNeedsReload(sessionId) {
      set((state) => ({
        sessionNeedsReload: {
          ...state.sessionNeedsReload,
          [sessionId]: true,
        },
      }));
    },
    clearNeedsReload(sessionId) {
      set((state) => {
        if (!state.sessionNeedsReload[sessionId]) return state;
        const sessionNeedsReload = { ...state.sessionNeedsReload };
        delete sessionNeedsReload[sessionId];
        return { sessionNeedsReload };
      });
    },
    resetAll() {
      set({ sessionNeedsReload: {} });
    },
  }),
);

const copilotChatRuntimes = new Map<string, CopilotChatRuntime>();

function markChatRuntimeDisconnected(sessionId: string) {
  useCopilotStreamStore.getState().clearSession(sessionId);
  useCopilotChatRuntimeStore.getState().markNeedsReload(sessionId);
}

function isTransientStreamDisconnect(error: Error) {
  const detail = error.message.toLowerCase();
  return (
    error.name === "TypeError" ||
    detail.includes("connection interrupted") ||
    detail.includes("network")
  );
}

export function getOrCreateCopilotChatRuntime(sessionId: string) {
  const existing = copilotChatRuntimes.get(sessionId);
  if (existing) return existing;

  const copilotModeRef: MutableValue<CopilotMode | undefined> = {
    current: undefined,
  };
  const copilotModelRef: MutableValue<CopilotLlmModel | undefined> = {
    current: undefined,
  };
  const callbacks: Pick<CopilotChatRuntime, "onFinish" | "onError"> = {};
  const chat = new Chat<UIMessage>({
    id: sessionId,
    transport: createCopilotTransport({
      sessionId,
      copilotModeRef,
      copilotModelRef,
    }),
    onFinish: (args) => {
      if (args.isDisconnect) {
        markChatRuntimeDisconnected(sessionId);
      }
      return callbacks.onFinish?.(args);
    },
    onError: (error) => {
      if (isTransientStreamDisconnect(error)) {
        markChatRuntimeDisconnected(sessionId);
      }
      return callbacks.onError?.(error);
    },
  });
  const runtime = {
    chat,
    copilotModeRef,
    copilotModelRef,
    get onFinish() {
      return callbacks.onFinish;
    },
    set onFinish(value) {
      callbacks.onFinish = value;
    },
    get onError() {
      return callbacks.onError;
    },
    set onError(value) {
      callbacks.onError = value;
    },
  } satisfies CopilotChatRuntime;

  copilotChatRuntimes.set(sessionId, runtime);
  return runtime;
}

export function markCopilotChatRuntimeHealthy(sessionId: string) {
  useCopilotChatRuntimeStore.getState().clearNeedsReload(sessionId);
}

export function shouldReloadCopilotChatRuntime(sessionId: string) {
  return !!useCopilotChatRuntimeStore.getState().sessionNeedsReload[sessionId];
}

export function resetCopilotChatRuntime(sessionId: string) {
  copilotChatRuntimes.delete(sessionId);
  useCopilotStreamStore.getState().clearSession(sessionId);
  useConnectedProvidersStore.getState().clearSession(sessionId);
  useCopilotChatRuntimeStore.getState().clearNeedsReload(sessionId);
}

export function resetCopilotChatRegistry() {
  copilotChatRuntimes.clear();
  useCopilotChatRuntimeStore.getState().resetAll();
}
