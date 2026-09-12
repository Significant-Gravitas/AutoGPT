import { useEffect, useRef } from "react";

export function useMCPRequest() {
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), []);

  function start() {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    return controller.signal;
  }

  function cancel() {
    active.current?.abort();
  }

  return { start, cancel };
}
