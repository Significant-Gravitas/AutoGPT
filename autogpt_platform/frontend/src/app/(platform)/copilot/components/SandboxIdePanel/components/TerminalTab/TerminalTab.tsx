"use client";

import "@xterm/xterm/css/xterm.css";
import { useTerminalTab } from "./useTerminalTab";

interface Props {
  sessionId: string;
}

export function TerminalTab({ sessionId }: Props) {
  const { containerRef, isClosed, reconnect } = useTerminalTab(sessionId);

  return (
    <div className="relative h-full min-h-0 bg-[#18181b]">
      <div ref={containerRef} className="h-full w-full p-2" />
      {isClosed ? (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50">
          <button
            type="button"
            onClick={reconnect}
            className="rounded bg-white px-3 py-1.5 text-sm font-medium text-zinc-800 hover:bg-zinc-100"
          >
            Reconnect
          </button>
        </div>
      ) : null}
    </div>
  );
}
