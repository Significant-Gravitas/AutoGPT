export type SheetSide = "top" | "right" | "bottom" | "left";

// Kobra fixes side panels at 440px through `data-[side]` variants; neutralise
// those so the house default and the caller's `className` set the width.
const SIDE_WIDTH =
  "w-3/4 data-[side=left]:w-auto data-[side=right]:w-auto sm:max-w-sm";

export function panelClassName(side: SheetSide) {
  return side === "left" || side === "right" ? SIDE_WIDTH : "";
}
