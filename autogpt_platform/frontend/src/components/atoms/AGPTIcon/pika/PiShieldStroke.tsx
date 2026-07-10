import React from "react";

/**
 * PiShieldStroke icon from the stroke style in security category.
 */
interface PiShieldStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiShieldStroke({
  size = 24,
  color,
  className,
  ariaLabel = "shield icon",
  ...props
}: PiShieldStrokeProps): JSX.Element {
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
        d="m5.496 4.314 5.388-1.946a3 3 0 0 1 2.038 0l5.465 1.974a3 3 0 0 1 1.972 2.591l.227 2.95A11 11 0 0 1 14.858 20.4l-1.49.806a3 3 0 0 1-2.914-.032l-1.52-.867A11 11 0 0 1 3.39 10.33l.127-3.31a3 3 0 0 1 1.98-2.705Z"
        fill="none"
      />
    </svg>
  );
}
