"use client";

import { useMiniGame } from "./useMiniGame";

export function MiniGame() {
  const { canvasRef } = useMiniGame();

  return (
    <div
      className="w-full overflow-hidden rounded-md bg-background text-foreground"
      style={{ border: "1px solid #d17fff" }}
    >
      <canvas
        ref={canvasRef}
        tabIndex={0}
        className="block w-full outline-none"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}
