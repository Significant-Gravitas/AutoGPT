import React from "react";

/**
 * PiArrowTurnUpRightStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiArrowTurnUpRightStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiArrowTurnUpRightStroke({
  size = 24,
  color,
  className,
  ariaLabel = "arrow-turn-up-right icon",
  ...props
}: PiArrowTurnUpRightStrokeProps): JSX.Element {
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
        d="M15.141 4a25.2 25.2 0 0 1 4.684 4.505A.8.8 0 0 1 20 9m-4.859 5a25.2 25.2 0 0 0 4.684-4.505A.8.8 0 0 0 20 9m0 0h-8c-2.8 0-4.2 0-5.27.545a5 5 0 0 0-2.185 2.185C4 12.8 4 14.2 4 17v3"
        fill="none"
      />
    </svg>
  );
}
