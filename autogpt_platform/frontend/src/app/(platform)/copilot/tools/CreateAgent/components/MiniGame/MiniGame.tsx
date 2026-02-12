"use client";

import { useMiniGame } from "./useMiniGame";

export function MiniGame() {
  const { canvasRef } = useMiniGame();

  return (
    <div className="w-full overflow-hidden rounded-md bg-background text-foreground">
      <canvas
        ref={canvasRef}
        className="block w-full"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}
