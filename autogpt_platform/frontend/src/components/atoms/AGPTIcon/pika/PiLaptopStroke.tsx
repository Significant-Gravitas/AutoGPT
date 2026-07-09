import React from "react";

/**
 * PiLaptopStroke icon from the stroke style in devices category.
 */
interface PiLaptopStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiLaptopStroke({
  size = 24,
  color,
  className,
  ariaLabel = "laptop icon",
  ...props
}: PiLaptopStrokeProps): JSX.Element {
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
        d="M21 16V8.8c0-1.68 0-2.52-.327-3.162a3 3 0 0 0-1.311-1.311C18.72 4 17.88 4 16.2 4H7.8c-1.68 0-2.52 0-3.162.327a3 3 0 0 0-1.311 1.311C3 6.28 3 7.12 3 8.8V16m18 0H3m18 0a1 1 0 0 1 1 1 3 3 0 0 1-3 3H5a3 3 0 0 1-3-3 1 1 0 0 1 1-1"
        fill="none"
      />
    </svg>
  );
}
