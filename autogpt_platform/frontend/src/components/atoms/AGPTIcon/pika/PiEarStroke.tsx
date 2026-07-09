import React from "react";

/**
 * PiEarStroke icon from the stroke style in medical category.
 */
interface PiEarStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiEarStroke({
  size = 24,
  color,
  className,
  ariaLabel = "ear icon",
  ...props
}: PiEarStrokeProps): JSX.Element {
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
        d="M13.526 2.666a6.41 6.41 0 0 1 2.968 12.089c-.685.358-1.028.537-1.135.644-.164.165-.131.115-.218.331-.056.14-.09.595-.16 1.503-.168 2.204-2.008 4.167-4.413 4.167a4.437 4.437 0 0 1-4.405-4.97c.264-2.21.954-5.129.954-7.355a6.41 6.41 0 0 1 6.409-6.409Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M16.532 9.058a2.975 2.975 0 1 0-5.951 0c0 .798-.046 1.552-.182 2.449 2.875.77 2.382 4.196-.37 3.82"
        fill="none"
      />
    </svg>
  );
}
