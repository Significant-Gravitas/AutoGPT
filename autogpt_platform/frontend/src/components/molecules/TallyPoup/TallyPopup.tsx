"use client";

import React from "react";
import { useTallyPopup } from "./useTallyPopup";
import { Button } from "@/components/atoms/Button/Button";

export function TallyPopupSimple() {
  const { state, handlers } = useTallyPopup();

  if (state.isFormVisible) {
    return null;
  }

  return (
    <div className="fixed bottom-1 right-0 z-20 hidden select-none items-center gap-4 p-3 transition-all duration-300 ease-in-out md:flex">
      {state.showTutorial && (
        <Button
          variant="primary"
          onClick={handlers.handleResetTutorial}
          className="mb-0 h-14 w-28 rounded-2xl bg-[rgba(65,65,64,1)] text-left font-sans text-lg font-medium leading-6"
        >
          Tutorial
        </Button>
      )}
      <Button
        variant="primary"
        data-tally-open="3yx2L0"
        data-tally-emoji-text="👋"
        data-tally-emoji-animation="wave"
        data-sentry-replay-id={state.sentryReplayId || "not-initialized"}
        data-sentry-replay-url={state.replayUrl || "not-initialized"}
        data-user-agent={state.userAgent}
        data-page-url={state.pageUrl}
        data-is-authenticated={
          state.isAuthenticated === null
            ? "unknown"
            : String(state.isAuthenticated)
        }
        data-email={state.userEmail || "not-authenticated"}
        className="mb-0 h-14 rounded-2xl bg-[rgba(65,65,64,1)] text-center font-sans text-lg font-medium leading-6"
      >
        Give Feedback
      </Button>
    </div>
  );
}

export default TallyPopupSimple;
