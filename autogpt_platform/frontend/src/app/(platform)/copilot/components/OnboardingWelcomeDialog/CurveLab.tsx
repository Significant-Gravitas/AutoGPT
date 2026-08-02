"use client";

import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import { useState, type PointerEvent, type RefObject } from "react";

export interface Point {
  x: number;
  y: number;
}

interface Tool {
  icon: string;
  x: number;
  y: number;
}

interface Props {
  stageRef: RefObject<HTMLDivElement | null>;
  orb: Point;
  tools: Tool[];
  controls: Record<string, Point>;
  onControlChange: (icon: string, control: Point) => void;
  curviness: number;
  onCurvinessChange: (curviness: number) => void;
}

// Testing-only curve lab: a pen-tool overlay for the orb wiring. Each arc
// gets a draggable control point (with handle lines back to the tile and
// the orb), the slider re-bows every arc at once, and "Copy curves" puts a
// ready-to-paste TOOLS snippet on the clipboard. Remove before release.
export function CurveLab({
  stageRef,
  orb,
  tools,
  controls,
  onControlChange,
  curviness,
  onCurvinessChange,
}: Props) {
  const [isCopied, setIsCopied] = useState(false);

  function handleDrag(icon: string) {
    return (event: PointerEvent<HTMLButtonElement>) => {
      if (event.buttons === 0) return;
      const stage = stageRef.current?.getBoundingClientRect();
      if (!stage) return;
      onControlChange(icon, {
        x: Math.round(((event.clientX - stage.left) / stage.width) * 1000) / 10,
        y: Math.round(((event.clientY - stage.top) / stage.height) * 1000) / 10,
      });
    };
  }

  function startDrag(event: PointerEvent<HTMLButtonElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  async function copyCurves() {
    const snippet = [
      "const TOOLS = [",
      ...tools.map((tool) => {
        const control = controls[tool.icon];
        return `  { icon: "${tool.icon}", x: ${tool.x}, y: ${tool.y}, cx: ${control.x}, cy: ${control.y} },`;
      }),
      "];",
    ].join("\n");
    await navigator.clipboard.writeText(snippet);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 1500);
  }

  return (
    <>
      <svg
        className="pointer-events-none absolute inset-0 z-20 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        fill="none"
      >
        {tools.map((tool) => {
          const control = controls[tool.icon];
          return (
            <g key={tool.icon}>
              <path
                d={`M ${tool.x} ${tool.y} L ${control.x} ${control.y} L ${orb.x} ${orb.y}`}
                stroke="#18181b"
                strokeWidth="1"
                strokeDasharray="3 3"
                strokeOpacity="0.35"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          );
        })}
      </svg>

      {tools.map((tool) => {
        const control = controls[tool.icon];
        return (
          <button
            key={tool.icon}
            type="button"
            aria-label={`Drag ${tool.icon} curve handle`}
            onPointerDown={startDrag}
            onPointerMove={handleDrag(tool.icon)}
            style={{ left: `${control.x}%`, top: `${control.y}%` }}
            className="absolute z-30 h-4 w-4 -translate-x-1/2 -translate-y-1/2 cursor-grab touch-none rounded-full border-2 border-white bg-violet-600 shadow active:cursor-grabbing"
          />
        );
      })}

      <div className="absolute bottom-3 left-1/2 z-30 flex w-72 -translate-x-1/2 items-center gap-2 rounded-full bg-zinc-900 px-3 py-1.5 text-xs font-medium text-white shadow-lg">
        <label htmlFor="curviness" className="shrink-0">
          Bow
        </label>
        <input
          id="curviness"
          type="range"
          min={-30}
          max={30}
          step={0.5}
          value={curviness}
          onChange={(event) => onCurvinessChange(Number(event.target.value))}
          className="h-1 w-full accent-violet-400"
        />
        <span className="w-8 shrink-0 text-right tabular-nums">
          {curviness}
        </span>
        <button
          type="button"
          onClick={copyCurves}
          className="flex shrink-0 items-center gap-1 rounded-full bg-white/15 px-2 py-1"
        >
          {isCopied ? (
            <CheckIcon size={12} weight="bold" className="text-emerald-400" />
          ) : (
            <CopyIcon size={12} />
          )}
          {isCopied ? "Copied" : "Copy"}
        </button>
      </div>
    </>
  );
}
