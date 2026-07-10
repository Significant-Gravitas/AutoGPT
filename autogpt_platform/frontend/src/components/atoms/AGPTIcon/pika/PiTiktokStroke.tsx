import React from "react";

/**
 * PiTiktokStroke icon from the stroke style in apps-&-social category.
 */
interface PiTiktokStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiTiktokStroke({
  size = 24,
  color,
  className,
  ariaLabel = "tiktok icon",
  ...props
}: PiTiktokStrokeProps): JSX.Element {
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
        d="M19.545 7.556a6.28 6.28 0 0 1-4.706-3.644L13.989 2v15.556a4.444 4.444 0 1 1-4.444-4.445"
        fill="none"
      />
    </svg>
  );
}
