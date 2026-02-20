"use client";

import { useMiniGame } from "./useMiniGame";

function Key({ children }: { children: React.ReactNode }) {
  return <strong>[{children}]</strong>;
}

export function MiniGame() {
  const { canvasRef, activeMode, showOverlay, score, highScore, onContinue } =
    useMiniGame();

  let overlayText: string | undefined;
  let buttonLabel = "Continue";
  if (activeMode === "idle") {
    buttonLabel = "Start";
  } else if (activeMode === "over") {
    overlayText = `Score: ${score} / Record: ${highScore}`;
    buttonLabel = "Retry";
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-medium text-purple-500">
        <Key>WASD</Key> to move
      </p>
      <div className="relative w-full overflow-hidden rounded-md border border-accent bg-background text-foreground">
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
