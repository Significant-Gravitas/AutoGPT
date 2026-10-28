"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import type { ScreenState } from "../useBrainDumpStep";

interface Props {
  screen: ScreenState;
  submittedMode: "voice" | "typed";
  onTypeInstead: () => void;
  onRecordInstead: () => void;
  onTypeInsteadOfRecovered: () => void;
}

// The quiet way across to the other mode, under the main controls.
export function ModeSwitchLink({
  screen,
  submittedMode,
  onTypeInstead,
  onRecordInstead,
  onTypeInsteadOfRecovered,
}: Props) {
  if (screen !== "typing" && screen !== "failed" && screen !== "recovery") {
    return null;
  }

  return (
    <div className="flex justify-center">
      <SwapFade swapKey={screen}>
        {screen === "failed" && (
          <LinkButton onClick={onTypeInstead}>
            {submittedMode === "typed" ? "edit your text" : "type instead"}
          </LinkButton>
        )}
        {/* Kept while the microphone is refused: it asks the browser again
            rather than going back to an orb that would only be refused. */}
        {screen === "typing" && (
          <LinkButton onClick={onRecordInstead}>record instead</LinkButton>
        )}
        {screen === "recovery" && (
          <LinkButton onClick={onTypeInsteadOfRecovered}>
            type instead
          </LinkButton>
        )}
      </SwapFade>
    </div>
  );
}

function LinkButton({
  onClick,
  children,
}: {
  onClick: () => void;
  children: string;
}) {
  return (
    <Button
      variant="ghost"
      size="xs"
      onClick={onClick}
      className="underline underline-offset-4"
    >
      {children}
    </Button>
  );
}
