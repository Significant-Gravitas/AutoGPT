import React from "react";

/**
 * PiBulbDefaultStroke icon from the stroke style in appliances category.
 */
interface PiBulbDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiBulbDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "bulb-default icon",
  ...props
}: PiBulbDefaultStrokeProps): JSX.Element {
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
        d="M10 22h4m-2-10v4m1-4h-2M5 9.737C5 6.017 8.134 3 12 3s7 3.016 7 6.737c0 2.04-.942 3.867-2.43 5.103-.628.521-1.168 1.17-1.373 1.96l-.28 1.077A1.5 1.5 0 0 1 13.465 19h-2.93a1.5 1.5 0 0 1-1.452-1.123l-.28-1.077c-.205-.79-.745-1.439-1.374-1.96C5.942 13.604 5 11.776 5 9.737Z"
        fill="none"
      />
    </svg>
  );
}
