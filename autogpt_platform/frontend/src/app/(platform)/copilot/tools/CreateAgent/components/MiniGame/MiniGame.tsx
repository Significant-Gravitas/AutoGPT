"use client";

import { useMiniGame } from "./useMiniGame";

function Key({ children }: { children: React.ReactNode }) {
  return <strong>[{children}]</strong>;
}

export function MiniGame() {
  const { canvasRef, activeMode, showOverlay, score, highScore, onContinue } =
    useMiniGame();

  const isRunActive =
    activeMode === "run" || activeMode === "idle" || activeMode === "over";
  const isBossActive =
    activeMode === "boss" ||
    activeMode === "boss-intro" ||
    activeMode === "boss-defeated";

  let overlayText: string | undefined;
  let buttonLabel = "Continue";
  if (activeMode === "idle") {
    buttonLabel = "Start";
  } else if (activeMode === "boss-intro") {
    overlayText = "Face the bandit!";
  } else if (activeMode === "boss-defeated") {
    overlayText = "Great job, keep on going";
  } else if (activeMode === "over") {
    overlayText = `Score: ${score} / Record: ${highScore}`;
    buttonLabel = "Retry";
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-medium text-purple-500">
        {isBossActive ? (
          <>
            Duel mode: <Key>←→</Key> to move · <Key>Z</Key> to attack ·{" "}
            <Key>X</Key> to block · <Key>Space</Key> to jump
          </>
        ) : (
          <>
            Run mode: <Key>Space</Key> to jump
          </>
        )}
      </p>
      <div
        className="relative w-full overflow-hidden rounded-md bg-background text-foreground"
        style={{ border: "1px solid #d17fff" }}
      >
        <canvas
          ref={canvasRef}
          tabIndex={0}
          className="block w-full outline-none"
        />
        {showOverlay && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black/40">
            {overlayText && (
              <p className="text-lg font-bold text-white">{overlayText}</p>
            )}
            <button
              type="button"
              onClick={onContinue}
              className="rounded-md bg-white px-4 py-2 text-sm font-semibold text-zinc-800 shadow-md transition-colors hover:bg-zinc-100"
            >
              {buttonLabel}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
