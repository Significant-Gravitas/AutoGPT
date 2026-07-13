import React from "react";

/**
 * PiInformationCircleStroke icon from the stroke style in alerts category.
 */
interface PiInformationCircleStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiInformationCircleStroke({
  size = 24,
  color,
  className,
  ariaLabel = "information-circle icon",
  ...props
}: PiInformationCircleStrokeProps): JSX.Element {
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
        d="M12 12v4m0-7.375zM21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z"
        fill="none"
      />
    </svg>
  );
}
