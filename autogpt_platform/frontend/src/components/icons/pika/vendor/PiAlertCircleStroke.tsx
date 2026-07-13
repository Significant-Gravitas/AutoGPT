import React from "react";

/**
 * PiAlertCircleStroke icon from the stroke style in alerts category.
 */
interface PiAlertCircleStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiAlertCircleStroke({
  size = 24,
  color,
  className,
  ariaLabel = "alert-circle icon",
  ...props
}: PiAlertCircleStrokeProps): JSX.Element {
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
        d="M12 12.624v-4M12 16zm9-4a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
        fill="none"
      />
    </svg>
  );
}
