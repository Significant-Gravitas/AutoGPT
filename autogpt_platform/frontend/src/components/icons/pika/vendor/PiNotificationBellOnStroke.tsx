import React from "react";

/**
 * PiNotificationBellOnStroke icon from the stroke style in alerts category.
 */
interface PiNotificationBellOnStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiNotificationBellOnStroke({
  size = 24,
  color,
  className,
  ariaLabel = "notification-bell-on icon",
  ...props
}: PiNotificationBellOnStrokeProps): JSX.Element {
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
        d="M9.159 17.724a60 60 0 0 1-3.733-.297 1.587 1.587 0 0 1-1.33-2.08c.161-.485.324-.963.367-1.478l.355-4.26a7.207 7.207 0 0 1 14.365 0l.355 4.262c.043.515.206.993.367 1.479a1.587 1.587 0 0 1-1.33 2.077 60 60 0 0 1-3.732.297m-5.684 0q2.841.135 5.684 0m-5.684 0v.434a2.842 2.842 0 1 0 5.684 0v-.434"
        fill="none"
      />
    </svg>
  );
}
