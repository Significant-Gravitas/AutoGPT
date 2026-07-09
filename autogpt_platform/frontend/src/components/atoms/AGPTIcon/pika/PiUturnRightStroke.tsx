import React from "react";

/**
 * PiUturnRightStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiUturnRightStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiUturnRightStroke({
  size = 24,
  color,
  className,
  ariaLabel = "uturn-right icon",
  ...props
}: PiUturnRightStrokeProps): JSX.Element {
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
        d="M15.97 3.916a20.8 20.8 0 0 1 3.885 3.679.64.64 0 0 1 .145.404m-4.03 4.084a20.8 20.8 0 0 0 3.885-3.68A.64.64 0 0 0 20 8m0 0H9a5 5 0 1 0 0 10h3"
        fill="none"
      />
    </svg>
  );
}
