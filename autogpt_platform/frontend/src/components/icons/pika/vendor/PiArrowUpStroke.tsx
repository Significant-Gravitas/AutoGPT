import React from "react";

/**
 * PiArrowUpStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiArrowUpStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiArrowUpStroke({
  size = 24,
  color,
  className,
  ariaLabel = "arrow-up icon",
  ...props
}: PiArrowUpStrokeProps): JSX.Element {
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
        d="M6 9.83a30.2 30.2 0 0 1 5.406-5.62A.95.95 0 0 1 12 4m6 5.83a30.2 30.2 0 0 0-5.406-5.62A.95.95 0 0 0 12 4m0 0v16"
        fill="none"
      />
    </svg>
  );
}
