import React from "react";

/**
 * PiGaugeSpeedometerStroke icon from the stroke style in general category.
 */
interface PiGaugeSpeedometerStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiGaugeSpeedometerStroke({
  size = 24,
  color,
  className,
  ariaLabel = "gauge-speedometer icon",
  ...props
}: PiGaugeSpeedometerStrokeProps): JSX.Element {
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
        d="M5.242 19.15a9.15 9.15 0 1 1 13.816 0M8.415 10.415l4.107 2.803a.938.938 0 1 1-1.304 1.305z"
        fill="none"
      />
    </svg>
  );
}
