import React from "react";

/**
 * PiNotebookStroke icon from the stroke style in general category.
 */
interface PiNotebookStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiNotebookStroke({
  size = 24,
  color,
  className,
  ariaLabel = "notebook icon",
  ...props
}: PiNotebookStrokeProps): JSX.Element {
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
        d="M12 5.066V21.5m0-16.434c3.025-1.1 7.005-1.71 10 0v15.5c-3.197-1.37-7.063-.401-10 .934m0-16.434c-3.025-1.1-7.005-1.71-10 0v15.5c3.197-1.37 7.063-.401 10 .934"
        fill="none"
      />
    </svg>
  );
}
