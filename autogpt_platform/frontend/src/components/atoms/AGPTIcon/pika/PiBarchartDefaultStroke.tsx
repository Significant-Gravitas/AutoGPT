import React from "react";

/**
 * PiBarchartDefaultStroke icon from the stroke style in chart-&-graph category.
 */
interface PiBarchartDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiBarchartDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "barchart-default icon",
  ...props
}: PiBarchartDefaultStrokeProps): JSX.Element {
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
        d="M9 21v-6c0-.932 0-1.398-.152-1.765a2 2 0 0 0-1.083-1.083C7.398 12 6.932 12 6 12s-1.398 0-1.765.152a2 2 0 0 0-1.083 1.083C3 13.602 3 14.068 3 15v2.8c0 1.12 0 1.68.218 2.108a2 2 0 0 0 .874.874C4.52 21 5.08 21 6.2 21zm0 0h6m-6 0V6c0-.932 0-1.398.152-1.765a2 2 0 0 1 1.083-1.083C10.602 3 11.068 3 12 3s1.398 0 1.765.152a2 2 0 0 1 1.083 1.083C15 4.602 15 5.068 15 6v15m0 0h2.8c1.12 0 1.68 0 2.108-.218a2 2 0 0 0 .874-.874C21 19.48 21 18.92 21 17.8V11c0-.932 0-1.398-.152-1.765a2 2 0 0 0-1.083-1.083C19.398 8 18.932 8 18 8s-1.398 0-1.765.152a2 2 0 0 0-1.083 1.083C15 9.602 15 10.068 15 11z"
        fill="none"
      />
    </svg>
  );
}
