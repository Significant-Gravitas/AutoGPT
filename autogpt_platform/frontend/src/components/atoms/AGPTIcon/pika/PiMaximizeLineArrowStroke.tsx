import React from "react";

/**
 * PiMaximizeLineArrowStroke icon from the stroke style in arrows-&-chevrons category.
 */
interface PiMaximizeLineArrowStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMaximizeLineArrowStroke({
  size = 24,
  color,
  className,
  ariaLabel = "maximize-line-arrow icon",
  ...props
}: PiMaximizeLineArrowStrokeProps): JSX.Element {
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
        d="M19.714 9.943a20.6 20.6 0 0 0 .2-5.296.62.62 0 0 0-.178-.383m-5.679.022a20.6 20.6 0 0 1 5.296-.2c.15.014.284.079.383.178m0 0L4.264 19.736m.022-5.679a20.6 20.6 0 0 0-.2 5.296c.014.15.079.284.178.383m5.679-.022a20.6 20.6 0 0 1-5.296.2.62.62 0 0 1-.383-.178"
        fill="none"
      />
    </svg>
  );
}
