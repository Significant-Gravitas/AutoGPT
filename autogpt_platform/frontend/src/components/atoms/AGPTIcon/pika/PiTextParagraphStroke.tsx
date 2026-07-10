import React from "react";

/**
 * PiTextParagraphStroke icon from the stroke style in editing category.
 */
interface PiTextParagraphStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiTextParagraphStroke({
  size = 24,
  color,
  className,
  ariaLabel = "text-paragraph icon",
  ...props
}: PiTextParagraphStrokeProps): JSX.Element {
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
        d="M13 3h5m-5 0H9.03a6.03 6.03 0 0 0 0 12.058H13M13 3v12.058M18 3v18m0-18h3m-8 18v-5.942"
        fill="none"
      />
    </svg>
  );
}
