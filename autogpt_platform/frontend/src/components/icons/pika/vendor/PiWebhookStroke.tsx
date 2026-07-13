import React from "react";

/**
 * PiWebhookStroke icon from the stroke style in development category.
 */
interface PiWebhookStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiWebhookStroke({
  size = 24,
  color,
  className,
  ariaLabel = "webhook icon",
  ...props
}: PiWebhookStrokeProps): JSX.Element {
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
        d="M15.996 6.494a4 4 0 1 0-5.916 3.512l-4.084 7.488m9.851 3.36a4 4 0 1 0 .244-6.875L12 6.495m-7.928 7.5a4 4 0 1 0 5.92 3.507l8.528-.008"
        fill="none"
      />
    </svg>
  );
}
