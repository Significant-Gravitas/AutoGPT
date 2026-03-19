"use client";

import { useTallyPopup } from "@/components/molecules/TallyPoup/useTallyPopup";
import { ChatCircleDotsIcon } from "@phosphor-icons/react";
import { motion } from "framer-motion";

export function FeedbackButton() {
  const { state } = useTallyPopup();

  if (state.isFormVisible) return null;

  return (
    <button
      type="button"
      className="group relative inline-flex overflow-hidden rounded-full active:scale-[0.97] disabled:pointer-events-none disabled:opacity-50"
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
      {/* Static gradient border (visible by default) */}
      <div className="absolute inset-0 rounded-full bg-gradient-to-r from-indigo-100 to-indigo-300 to-zinc-400 transition-opacity duration-300 group-hover:opacity-0" />

      {/* Animated spinning gradient (visible on hover) */}
      <motion.div
        className="absolute opacity-0 transition-opacity duration-300 group-hover:opacity-100"
        animate={{ rotate: 360 }}
        transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
        style={{
          inset: "-50%",
          background:
            "conic-gradient(from 0deg, #6366f1, #818cf8, #a78bfa, #7c3aed, #6366f1)",
        }}
      />

      {/* Inner content */}
      <div className="relative m-[1px] flex items-center gap-1.5 rounded-full bg-[#FAFAFA]/80 px-3 py-1.5 text-sm font-medium text-neutral-700 backdrop-blur-xl transition-colors duration-150 ease-out group-hover:bg-zinc-50/90">
        Give Feedback
        <ChatCircleDotsIcon size={16} />
      </div>
    </button>
  );
}
