import React from "react";

/**
 * PiShare01Stroke icon from the stroke style in general category.
 */
interface PiShare01StrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiShare01Stroke({
  size = 24,
  color,
  className,
  ariaLabel = "share-01 icon",
  ...props
}: PiShare01StrokeProps): JSX.Element {
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
        d="M15.41 6.51c-2.583.773-4.924 2.033-6.82 3.98m6.82 7c-2.583-.773-4.925-2.033-6.82-3.98M21 5a3 3 0 1 1-6 0 3 3 0 0 1 6 0ZM9 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm12 7a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
        fill="none"
      />
    </svg>
  );
}
