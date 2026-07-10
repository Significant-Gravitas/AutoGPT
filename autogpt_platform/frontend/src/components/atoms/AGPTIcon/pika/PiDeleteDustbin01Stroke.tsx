import React from "react";

/**
 * PiDeleteDustbin01Stroke icon from the stroke style in general category.
 */
interface PiDeleteDustbin01StrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiDeleteDustbin01Stroke({
  size = 24,
  color,
  className,
  ariaLabel = "delete-dustbin-01 icon",
  ...props
}: PiDeleteDustbin01StrokeProps): JSX.Element {
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
        d="m16 6-1.106-2.211a3.236 3.236 0 0 0-5.788 0L8 6M4 6h16M6 6h12v9c0 1.864 0 2.796-.305 3.53a4 4 0 0 1-2.164 2.165C14.796 21 13.864 21 12 21s-2.796 0-3.53-.305a4 4 0 0 1-2.166-2.164C6 17.796 6 16.864 6 15z"
        fill="none"
      />
    </svg>
  );
}
