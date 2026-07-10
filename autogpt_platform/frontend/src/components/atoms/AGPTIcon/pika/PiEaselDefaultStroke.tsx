import React from "react";

/**
 * PiEaselDefaultStroke icon from the stroke style in general category.
 */
interface PiEaselDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiEaselDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "easel-default icon",
  ...props
}: PiEaselDefaultStrokeProps): JSX.Element {
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
        d="M12 3H8.4c-2.24 0-3.36 0-4.216.436a4 4 0 0 0-1.748 1.748C2 6.04 2 7.16 2 9.4v2.2c0 2.24 0 3.36.436 4.216a4 4 0 0 0 1.748 1.748c.803.41 1.84.434 3.816.436m4-15h3.6c2.24 0 3.36 0 4.216.436a4 4 0 0 1 1.748 1.748C22 6.04 22 7.16 22 9.4v2.2c0 2.24 0 3.36-.436 4.216a4 4 0 0 1-1.748 1.748c-.799.407-1.829.434-3.785.436M12 3V2M6 22l2-4m10 4-1.97-4M12 21v-2.998M16.03 18l-4.03.002M8 18l4 .002"
        fill="none"
      />
    </svg>
  );
}
