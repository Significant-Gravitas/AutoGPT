import type { IconSvgElement } from "@hugeicons/react";

// Glyphs Hugeicons has no exact match for, drawn as icon data so they render
// through the Icon atom exactly as their old hand-written SVGs did. Swapping
// them for Hugeicons glyphs changes what renders, so that is a design call.

const ROUND = { strokeLinecap: "round", strokeLinejoin: "round" } as const;

export const CircleAlertGlyph: IconSvgElement = [
  ["circle", { cx: "12", cy: "12", r: "10", ...ROUND, key: "0" }],
  ["line", { x1: "12", x2: "12", y1: "8", y2: "12", ...ROUND, key: "1" }],
  ["line", { x1: "12", x2: "12.01", y1: "16", y2: "16", ...ROUND, key: "2" }],
];

const STAR_POINTS =
  "12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2";

export const StarGlyph: IconSvgElement = [
  ["polygon", { points: STAR_POINTS, ...ROUND, key: "0" }],
];

export const StarFilledGlyph: IconSvgElement = [
  [
    "polygon",
    { points: STAR_POINTS, fill: "currentColor", ...ROUND, key: "0" },
  ],
];

// Drawn at strokeWidth={1.25}.
export const DashboardLayoutGlyph: IconSvgElement = [
  ["path", { d: "M10 3H3V12H10V3Z", ...ROUND, key: "0" }],
  ["path", { d: "M21 3H14V8H21V3Z", ...ROUND, key: "1" }],
  ["path", { d: "M21 12H14V21H21V12Z", ...ROUND, key: "2" }],
  ["path", { d: "M10 16H3V21H10V16Z", ...ROUND, key: "3" }],
];

// Drawn on a 20px grid: render with viewBox="0 0 20 20" and strokeWidth={1.5}.
export const MoreVerticalGlyph: IconSvgElement = [
  [
    "path",
    {
      d: "M10 10.8333C10.4603 10.8333 10.8334 10.4602 10.8334 9.99999C10.8334 9.53975 10.4603 9.16666 10 9.16666C9.53978 9.16666 9.16669 9.53975 9.16669 9.99999C9.16669 10.4602 9.53978 10.8333 10 10.8333Z",
      fill: "currentColor",
      ...ROUND,
      key: "0",
    },
  ],
  [
    "path",
    {
      d: "M10 4.99999C10.4603 4.99999 10.8334 4.6269 10.8334 4.16666C10.8334 3.70642 10.4603 3.33333 10 3.33333C9.53978 3.33333 9.16669 3.70642 9.16669 4.16666C9.16669 4.6269 9.53978 4.99999 10 4.99999Z",
      fill: "currentColor",
      ...ROUND,
      key: "1",
    },
  ],
  [
    "path",
    {
      d: "M10 16.6667C10.4603 16.6667 10.8334 16.2936 10.8334 15.8333C10.8334 15.3731 10.4603 15 10 15C9.53978 15 9.16669 15.3731 9.16669 15.8333C9.16669 16.2936 9.53978 16.6667 10 16.6667Z",
      fill: "currentColor",
      ...ROUND,
      key: "2",
    },
  ],
];
