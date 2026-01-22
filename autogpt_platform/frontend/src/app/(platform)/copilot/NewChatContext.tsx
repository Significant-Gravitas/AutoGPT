"use client";

import { createContext, useContext, useRef, type ReactNode } from "react";

interface NewChatContextValue {
  onNewChatClick: () => void;
  setOnNewChatClick: (handler?: () => void) => void;
  performNewChat?: () => void;
  setPerformNewChat: (handler?: () => void) => void;
}

const NewChatContext = createContext<NewChatContextValue | null>(null);

export function NewChatProvider({ children }: { children: ReactNode }) {
  const onNewChatRef = useRef<(() => void) | undefined>();
  const performNewChatRef = useRef<(() => void) | undefined>();
  const contextValueRef = useRef<NewChatContextValue>({
    onNewChatClick() {
      onNewChatRef.current?.();
    },
    setOnNewChatClick(handler?: () => void) {
      onNewChatRef.current = handler;
    },
    performNewChat() {
      performNewChatRef.current?.();
    },
    setPerformNewChat(handler?: () => void) {
      performNewChatRef.current = handler;
    },
  });

  return (
    <NewChatContext.Provider value={contextValueRef.current}>
      {children}
    </NewChatContext.Provider>
  );
}

export function useNewChat() {
  return useContext(NewChatContext);
}
