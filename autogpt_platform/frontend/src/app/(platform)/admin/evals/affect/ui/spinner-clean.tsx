"use client";

import type { ComponentPropsWithRef } from "react";

/*
 * Spinner – a single self-contained file, clean style.
 * A small arc turning on a faint ring, drawn in the text color (currentColor), for something that works right now:
 * a run, a check, a connect. With `label` it is announced as a status; without it stays silent next to text that
 * already says what happens. Size it with `size`, color it with `color` on it or any parent.
 * The styles are injected below, so no extra files are needed.
 */

export type SpinnerProps = Omit<
  ComponentPropsWithRef<"svg">,
  "children" | "width" | "height" | "viewBox"
> & {
  /** Width and height in px */
  size?: number;
  /** Announced to screen readers as a status – leave it out when text next to it already says what happens */
  label?: string;
};

export function Spinner({
  size = 14,
  label,
  className,
  ...rest
}: SpinnerProps) {
  return (
    <>
      {/* React 19 hoists this into <head> once, no matter how many spinners render */}
      <style href="ui-spinner-clean" precedence="default">
        {css}
      </style>
      <svg
        {...rest}
        className={["ui-spnc", className].filter(Boolean).join(" ")}
        viewBox="0 0 16 16"
        width={size}
        height={size}
        {...(label
          ? { role: "status", "aria-label": label }
          : { "aria-hidden": true })}
      >
        <circle
          cx="8"
          cy="8"
          r="6"
          fill="none"
          stroke="currentColor"
          strokeOpacity="0.18"
          strokeWidth="1.8"
        />
        <path
          d="M8 2a6 6 0 0 1 6 6"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
        />
      </svg>
    </>
  );
}

const css = /* css */ `
/* One turn every --ui-spnc-speed – override it on .ui-spnc (see Theming in the docs) */
.ui-spnc {
  --ui-spnc-speed: 800ms;
  display: inline-block;
  flex: none;
  vertical-align: middle;
  animation: ui-spnc-turn var(--ui-spnc-speed) linear infinite;
}
@keyframes ui-spnc-turn { to { rotate: 360deg; } }

/* Reduced motion: it still turns – it is the only sign that work goes on – but slowly */
@media (prefers-reduced-motion: reduce) {
  .ui-spnc { --ui-spnc-speed: 2400ms; }
}
`;
