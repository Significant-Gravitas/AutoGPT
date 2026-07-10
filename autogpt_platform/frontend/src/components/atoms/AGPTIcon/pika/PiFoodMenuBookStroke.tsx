import React from "react";

/**
 * PiFoodMenuBookStroke icon from the stroke style in food category.
 */
interface PiFoodMenuBookStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFoodMenuBookStroke({
  size = 24,
  color,
  className,
  ariaLabel = "food-menu-book icon",
  ...props
}: PiFoodMenuBookStrokeProps): JSX.Element {
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
        d="M16 12v-1a4 4 0 0 0-4-4m4 5H8m8 0h1m-9 0v-1a4 4 0 0 1 4-4m-4 5H7m5-5V6M2 8h2m-2 8h2m5 0h6m-4 6h2c2.8 0 4.2 0 5.27-.545a5 5 0 0 0 2.185-2.185C21 18.2 21 16.8 21 14v-4c0-2.8 0-4.2-.545-5.27a5 5 0 0 0-2.185-2.185C17.2 2 15.8 2 13 2h-2c-2.8 0-4.2 0-5.27.545A5 5 0 0 0 3.545 4.73C3 5.8 3 7.2 3 10v4c0 2.8 0 4.2.545 5.27a5 5 0 0 0 2.185 2.185C6.8 22 8.2 22 11 22Z"
        fill="none"
      />
    </svg>
  );
}
