import React from "react";

/**
 * PiTwitterStroke icon from the stroke style in apps-&-social category.
 */
interface PiTwitterStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiTwitterStroke({
  size = 24,
  color,
  className,
  ariaLabel = "twitter icon",
  ...props
}: PiTwitterStrokeProps): JSX.Element {
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
        d="M20.96 5.255c.18-.41-.29-.756-.686-.545-.618.332-1.27.602-1.944.805-2.714-3.39-7.39-.536-6.699 3.12.022.118-.066.233-.187.23-2.542-.047-4.337-.874-6.069-2.823-.246-.277-.681-.264-.867.056-1.144 1.969-3.97 8.074 3.298 10.523-1.421.964-3.275 1.784-4.225 2.175-.235.097-.245.43-.014.535 9.484 4.272 18.713-1.95 15.79-11.742a7.5 7.5 0 0 0 1.604-2.334Z"
        fill="none"
      />
    </svg>
  );
}
