import React from "react";

/**
 * PiPencilEditBoxStroke icon from the stroke style in general category.
 */
interface PiPencilEditBoxStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiPencilEditBoxStroke({
  size = 24,
  color,
  className,
  ariaLabel = "pencil-edit-box icon",
  ...props
}: PiPencilEditBoxStrokeProps): JSX.Element {
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
        d="M20 14c0 2.8 0 4.2-.545 5.27a5 5 0 0 1-2.185 2.185C16.2 22 14.8 22 12 22h-2c-2.8 0-4.2 0-5.27-.545a5 5 0 0 1-2.185-2.185C2 18.2 2 16.8 2 14v-2c0-2.8 0-4.2.545-5.27A5 5 0 0 1 4.73 4.545C5.8 4 7.2 4 10 4m-1.938 9.502c.008-.351.013-.527.055-.691q.057-.22.177-.414c.09-.144.213-.268.46-.517l9.396-9.45a1.46 1.46 0 0 1 1.828-.196A5.9 5.9 0 0 1 21.7 3.946l.075.116c.376.605.253 1.371-.24 1.867l-9.322 9.375c-.257.257-.385.386-.534.478a1.5 1.5 0 0 1-.429.178c-.17.04-.351.04-.714.04L8 15.996z"
        fill="none"
      />
    </svg>
  );
}
