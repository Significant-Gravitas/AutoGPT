import React from "react";

/**
 * PiNotificationBellOffStroke icon from the stroke style in alerts category.
 */
interface PiNotificationBellOffStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiNotificationBellOffStroke({
  size = 24,
  color,
  className,
  ariaLabel = "notification-bell-off icon",
  ...props
}: PiNotificationBellOffStrokeProps): JSX.Element {
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
        d="M6.466 17.534 3 21m3.466-3.466L17.914 6.086M6.466 17.534a60 60 0 0 1-1.04-.107 1.587 1.587 0 0 1-1.33-2.08c.161-.485.324-.963.367-1.478l.355-4.26a7.207 7.207 0 0 1 13.096-3.523M21 3l-3.086 3.086m1.285 3.71.34 4.075c.042.515.205.993.366 1.479a1.587 1.587 0 0 1-1.33 2.077 60 60 0 0 1-7.366.36L9.495 19.5a2.842 2.842 0 0 0 5.348-1.342v-.346"
        fill="none"
      />
    </svg>
  );
}
