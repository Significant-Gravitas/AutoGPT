import React from "react";

/**
 * PiTimerDefaultStroke icon from the stroke style in time category.
 */
interface PiTimerDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiTimerDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "timer-default icon",
  ...props
}: PiTimerDefaultStrokeProps): JSX.Element {
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
        d="M12 2v4m0 0a8 8 0 1 0 0 16 8 8 0 0 0 0-16Zm-2-4h4m5.366 3.322 1.06 1.06M14.639 11.1l-2.829 2.83m2.842-2.83-2.829 2.828"
        fill="none"
      />
    </svg>
  );
}
