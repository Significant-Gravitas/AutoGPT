import React from "react";

/**
 * PiRotateRightStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiRotateRightStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiRotateRightStroke({
  size = 24,
  color,
  className,
  ariaLabel = "rotate-right icon",
  ...props
}: PiRotateRightStrokeProps): JSX.Element {
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
        d="M17.783 2.67a15 15 0 0 1 1.049 3.726c.049.335-.215.485-.479.586l-.094.035m0 0A8 8 0 1 0 19.748 14m-1.489-6.983a15 15 0 0 1-3.476.85"
        fill="none"
      />
    </svg>
  );
}
