import React from "react";

/**
 * PiPlayBigStroke icon from the stroke style in media category.
 */
interface PiPlayBigStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiPlayBigStroke({
  size = 24,
  color,
  className,
  ariaLabel = "play-big icon",
  ...props
}: PiPlayBigStrokeProps): JSX.Element {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      style={{ color: color || "currentColor" }}
      role="img"
      aria-label={ariaLabel}
      {...props}
    >
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M5 11.723c0-3.818 0-5.728.798-6.793a4 4 0 0 1 2.917-1.593c1.328-.095 2.934.938 6.146 3.002l.431.278c2.787 1.791 4.18 2.687 4.662 3.826a4 4 0 0 1 0 3.114c-.481 1.14-1.875 2.035-4.662 3.827l-.431.277c-3.212 2.065-4.818 3.097-6.146 3.002a4 4 0 0 1-2.917-1.592C5 18.005 5 16.096 5 12.278z"
        fill="none"
      />
    </svg>
  );
}
