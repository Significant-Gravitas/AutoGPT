import React from "react";

/**
 * PiMediaPlaySquareStroke icon from the stroke style in media category.
 */
interface PiMediaPlaySquareStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMediaPlaySquareStroke({
  size = 24,
  color,
  className,
  ariaLabel = "media-play-square icon",
  ...props
}: PiMediaPlaySquareStrokeProps): JSX.Element {
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
        strokeLinejoin="round"
        strokeWidth="2"
        d="M4 9.4c0-2.24 0-3.36.436-4.216a4 4 0 0 1 1.748-1.748C7.04 3 8.16 3 10.4 3h3.2c2.24 0 3.36 0 4.216.436a4 4 0 0 1 1.748 1.748C20 6.04 20 7.16 20 9.4v5.2c0 2.24 0 3.36-.436 4.216a4 4 0 0 1-1.748 1.748C16.96 21 15.84 21 13.6 21h-3.2c-2.24 0-3.36 0-4.216-.436a4 4 0 0 1-1.748-1.748C4 17.96 4 16.84 4 14.6z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M13.114 9.89c1.199.692 1.799 1.039 2 1.49a1.53 1.53 0 0 1 0 1.24c-.201.451-.801.798-2 1.49l-.257.148c-1.2.693-1.799 1.039-2.29.987-.43-.045-.82-.27-1.074-.62-.29-.4-.29-1.092-.29-2.477v-.296c0-1.385 0-2.077.29-2.478a1.52 1.52 0 0 1 1.073-.619c.492-.052 1.092.295 2.291.987z"
        fill="none"
      />
    </svg>
  );
}
