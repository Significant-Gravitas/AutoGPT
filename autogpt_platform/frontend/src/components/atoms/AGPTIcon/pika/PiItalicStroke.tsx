import React from "react";

/**
 * PiItalicStroke icon from the stroke style in editing category.
 */
interface PiItalicStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiItalicStroke({
  size = 24,
  color,
  className,
  ariaLabel = "italic icon",
  ...props
}: PiItalicStrokeProps): JSX.Element {
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
        d="m13.5 5-3 14m3-14H17m-3.5 0H10m.5 14H14m-3.5 0H7"
        fill="none"
      />
    </svg>
  );
}
