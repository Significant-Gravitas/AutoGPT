"use client";

import React from "react";
import { useTallyPopup } from "./useTallyPopup";
import { Button } from "@/components/atoms/Button/Button";
import { usePathname, useSearchParams } from "next/navigation";

export function TallyPopupSimple() {
  const { state, handlers } = useTallyPopup();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const isNewBuilder =
    pathname.includes("build") && searchParams.get("view") === "new";

  if (state.isFormVisible) {
    return null;
  }

  return (
    <div className="fixed right-0 bottom-1 z-20 hidden items-center gap-4 p-3 transition-all duration-300 ease-in-out select-none md:flex">
      {state.showTutorial && !isNewBuilder && (
        <Button
          variant="primary"
          onClick={handlers.handleResetTutorial}
          className="mb-0 h-14 w-28 rounded-2xl bg-[rgba(65,65,64,1)] text-left font-sans text-lg leading-6 font-medium"
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
        className="mb-0 h-14 rounded-2xl bg-[rgba(65,65,64,1)] text-center font-sans text-lg leading-6 font-medium"
      >
        Give Feedback
      </Button>
    </div>
  );
}

export default TallyPopupSimple;
