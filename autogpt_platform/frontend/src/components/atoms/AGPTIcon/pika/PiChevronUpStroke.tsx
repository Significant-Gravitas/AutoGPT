import React from "react";

/**
 * PiChevronUpStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiChevronUpStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiChevronUpStroke({
  size = 24,
  color,
  className,
  ariaLabel = "chevron-up icon",
  ...props
}: PiChevronUpStrokeProps): JSX.Element {
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
        d="M8 14a20.4 20.4 0 0 1 3.702-3.894.47.47 0 0 1 .596 0A20.4 20.4 0 0 1 16 14"
        fill="none"
      />
    </svg>
  );
}
