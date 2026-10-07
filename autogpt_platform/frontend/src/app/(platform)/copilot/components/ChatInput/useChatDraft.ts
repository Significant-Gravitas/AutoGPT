import { useEffect, useRef, useState, type SetStateAction } from "react";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

const subscribers = new Map<string, Set<(value: string) => void>>();

function readDraft(key: string | null) {
  if (!key || typeof window === "undefined") return "";
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function writeDraft(key: string | null, value: string) {
  if (!key || typeof window === "undefined") return;
  try {
    if (value) sessionStorage.setItem(key, value);
    else sessionStorage.removeItem(key);
  } catch {
    /* Keep the in-memory draft when browser storage is unavailable. */
  }
  subscribers.get(key)?.forEach((receive) => receive(value));
}

export function useChatDraft(draftKey?: string) {
  const userID = useAuthStore((state) => state.user?.id);
  const key =
    draftKey && userID
      ? `autogpt:chat-draft:${encodeURIComponent(userID)}:${encodeURIComponent(draftKey)}`
      : null;
  const [snapshot, setSnapshot] = useState(() => ({
    key,
    value: readDraft(key),
  }));
  const visible =
    snapshot.key === key ? snapshot : { key, value: readDraft(key) };
  const current = useRef(visible);
  current.current = visible;
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  useEffect(() => {
    if (!key) return;
    function receive(value: string) {
      const next = { key, value };
      current.current = next;
      setSnapshot(next);
    }
    const listeners = subscribers.get(key) ?? new Set();
    listeners.add(receive);
    subscribers.set(key, listeners);
    return () => {
      listeners.delete(receive);
      if (!listeners.size) subscribers.delete(key);
    };
  }, [key]);
  function setValue(update: SetStateAction<string>) {
    const previous =
      mounted.current && current.current.key === key
        ? current.current.value
        : readDraft(key);
    const value = typeof update === "function" ? update(previous) : update;
    writeDraft(key, value);
    if (mounted.current && current.current.key === key) {
      const next = { key, value };
      current.current = next;
      setSnapshot(next);
    }
  }
  return [visible.value, setValue] as const;
}
