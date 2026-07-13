import React from "react";

/**
 * PiChevronSortVerticalStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiChevronSortVerticalStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiChevronSortVerticalStroke({
  size = 24,
  color,
  className,
  ariaLabel = "chevron-sort-vertical icon",
  ...props
}: PiChevronSortVerticalStrokeProps): JSX.Element {
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
        d="M16 9a20.4 20.4 0 0 0-3.702-3.894.47.47 0 0 0-.596 0A20.4 20.4 0 0 0 8 9m0 6a20.4 20.4 0 0 0 3.702 3.894c.175.141.42.141.596 0A20.4 20.4 0 0 0 16 15"
        fill="none"
      />
    </svg>
  );
}
