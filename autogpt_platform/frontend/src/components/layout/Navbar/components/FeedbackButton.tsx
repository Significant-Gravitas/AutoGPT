"use client";

import { useTallyPopup } from "@/components/molecules/TallyPoup/useTallyPopup";
import { ChatCircleDotsIcon } from "@phosphor-icons/react";

export function FeedbackButton() {
  const { state } = useTallyPopup();

  if (state.isFormVisible) return null;

  return (
    <button
      type="button"
      className="group inline-flex overflow-hidden rounded-full active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50"
      data-tally-open="3yx2L0"
      data-tally-emoji-text="👋"
      data-tally-emoji-animation="wave"
      data-sentry-replay-id={state.sentryReplayId || "not-initialized"}
      data-sentry-replay-url={state.replayUrl || "not-initialized"}
      data-page-url={
        state.pageUrl ? state.pageUrl.split("?")[0] : "not-initialized"
      }
      data-is-authenticated={
        state.isAuthenticated === null
          ? "unknown"
          : String(state.isAuthenticated)
      }
    >
      <div className="rounded-full bg-gradient-to-r from-indigo-100 to-indigo-300 to-zinc-400 p-[1px]">
        <div className="flex items-center gap-1.5 whitespace-nowrap rounded-full bg-[#FAFAFA]/80 px-3 py-1.5 text-sm font-medium text-neutral-700 backdrop-blur-xl transition-colors duration-150 ease-out group-hover:bg-zinc-100/90">
          <span className="hidden xl:inline">Give Feedback</span>
          <ChatCircleDotsIcon size={16} />
        </div>
      </div>
    </button>
  );
}
