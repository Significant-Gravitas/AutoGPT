import React from "react";

/**
 * PiArrowRightStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiArrowRightStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiArrowRightStroke({
  size = 24,
  color,
  className,
  ariaLabel = "arrow-right icon",
  ...props
}: PiArrowRightStrokeProps): JSX.Element {
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
        d="M15.17 6a30.2 30.2 0 0 1 5.62 5.406c.14.174.21.384.21.594m-5.83 6a30.2 30.2 0 0 0 5.62-5.406A.95.95 0 0 0 21 12m0 0H3"
        fill="none"
      />
    </svg>
  );
}
