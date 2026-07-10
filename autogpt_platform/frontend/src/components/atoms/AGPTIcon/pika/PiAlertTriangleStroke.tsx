import React from "react";

/**
 * PiAlertTriangleStroke icon from the stroke style in alerts category.
 */
interface PiAlertTriangleStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiAlertTriangleStroke({
  size = 24,
  color,
  className,
  ariaLabel = "alert-triangle icon",
  ...props
}: PiAlertTriangleStrokeProps): JSX.Element {
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
        d="M12 13V9m0 7.375v.001M10.61 3.284a3.55 3.55 0 0 1 2.78 0c2.651 1.128 8.915 11.138 8.731 13.813a3.63 3.63 0 0 1-1.424 2.645c-2.212 1.677-15.182 1.677-17.394 0a3.63 3.63 0 0 1-1.424-2.645c-.184-2.675 6.08-12.685 8.731-13.813Z"
        fill="none"
      />
    </svg>
  );
}
