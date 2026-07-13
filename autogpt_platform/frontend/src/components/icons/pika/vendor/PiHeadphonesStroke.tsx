import React from "react";

/**
 * PiHeadphonesStroke icon from the stroke style in media category.
 */
interface PiHeadphonesStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiHeadphonesStroke({
  size = 24,
  color,
  className,
  ariaLabel = "headphones icon",
  ...props
}: PiHeadphonesStrokeProps): JSX.Element {
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
        d="M20.993 15.618a2.378 2.378 0 0 0-4.572-1.31l-1.049 3.658a2.378 2.378 0 1 0 4.572 1.31zm0 0a9.5 9.5 0 0 0 .519-3.106 9.512 9.512 0 1 0-19.024 0c0 1.088.182 2.132.518 3.105m17.987.001-.007.022m-17.98-.023a2.378 2.378 0 0 1 4.573-1.309l1.048 3.658a2.378 2.378 0 0 1-4.571 1.31zm0 0 .018.052"
        fill="none"
      />
    </svg>
  );
}
