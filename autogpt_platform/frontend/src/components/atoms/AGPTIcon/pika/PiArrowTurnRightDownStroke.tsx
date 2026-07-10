import React from "react";

/**
 * PiArrowTurnRightDownStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiArrowTurnRightDownStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiArrowTurnRightDownStroke({
  size = 24,
  color,
  className,
  ariaLabel = "arrow-turn-right-down icon",
  ...props
}: PiArrowTurnRightDownStrokeProps): JSX.Element {
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
        d="M20 15.141a25.2 25.2 0 0 1-4.505 4.684A.8.8 0 0 1 15 20m-5-4.859a25.2 25.2 0 0 0 4.505 4.684A.8.8 0 0 0 15 20m0 0v-8c0-2.8 0-4.2-.545-5.27a5 5 0 0 0-2.185-2.185C11.2 4 9.8 4 7 4H4"
        fill="none"
      />
    </svg>
  );
}
