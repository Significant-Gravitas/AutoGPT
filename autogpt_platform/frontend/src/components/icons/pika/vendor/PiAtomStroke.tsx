import React from "react";

/**
 * PiAtomStroke icon from the stroke style in general category.
 */
interface PiAtomStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiAtomStroke({
  size = 24,
  color,
  className,
  ariaLabel = "atom icon",
  ...props
}: PiAtomStrokeProps): JSX.Element {
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
        d="M12 18.987c-3.36 2.116-6.583 2.678-8.124 1.137-1.54-1.541-.98-4.764 1.137-8.124M12 18.987a22.5 22.5 0 0 0 3.868-3.118A22.5 22.5 0 0 0 18.988 12M12 18.987c3.36 2.116 6.583 2.678 8.124 1.137 1.54-1.541.98-4.764-1.137-8.124M12 18.987a22.5 22.5 0 0 1-3.868-3.118A22.4 22.4 0 0 1 5.013 12m13.974 0c2.116-3.36 2.678-6.583 1.137-8.124-1.541-1.54-4.764-.98-8.124 1.137M18.987 12a22.5 22.5 0 0 0-3.118-3.868A22.5 22.5 0 0 0 12 5.012m0 0a22.5 22.5 0 0 0-3.868 3.12A22.4 22.4 0 0 0 5.013 12M12 5.013C8.64 2.897 5.417 2.335 3.876 3.876 2.336 5.417 2.896 8.64 5.013 12M11 12a1 1 0 1 1 2 0 1 1 0 0 1-2 0Z"
        fill="none"
      />
    </svg>
  );
}
