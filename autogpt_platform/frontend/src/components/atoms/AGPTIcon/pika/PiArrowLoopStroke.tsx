import React from "react";

/**
 * PiArrowLoopStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiArrowLoopStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiArrowLoopStroke({
  size = 24,
  color,
  className,
  ariaLabel = "arrow-loop icon",
  ...props
}: PiArrowLoopStrokeProps): JSX.Element {
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
        d="M16.223 17.629q.371-.324.727-.679c3.905-3.905 4.855-9.287 2.121-12.02-1.832-1.833-4.854-2.01-7.823-.753M7.786 6.364q-.375.327-.736.687c-3.905 3.905-4.855 9.287-2.121 12.02 1.832 1.832 4.854 2.01 7.823.752"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M3.652 6.205a15 15 0 0 1 3.685-.069.7.7 0 0 1 .627.628 15 15 0 0 1-.069 3.684m8.218 3.096a15 15 0 0 0-.07 3.684.7.7 0 0 0 .628.627 15 15 0 0 0 3.685-.069"
        fill="none"
      />
    </svg>
  );
}
