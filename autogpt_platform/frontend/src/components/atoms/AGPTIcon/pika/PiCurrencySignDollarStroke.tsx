import React from "react";

/**
 * PiCurrencySignDollarStroke icon from the stroke style in money-&-payments category.
 */
interface PiCurrencySignDollarStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiCurrencySignDollarStroke({
  size = 24,
  color,
  className,
  ariaLabel = "currency-sign-dollar icon",
  ...props
}: PiCurrencySignDollarStrokeProps): JSX.Element {
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
        d="M12 3v1.8m0 0V12m0-7.2h-1.667C8.493 4.8 7 6.412 7 8.4S8.492 12 10.333 12H12m0-7.2h1.772c1.553 0 2.858 1.147 3.228 2.7M12 12v7.2m0-7.2h1.667C15.507 12 17 13.612 17 15.6s-1.492 3.6-3.333 3.6H12m0 0V21m0-1.8h-1.772C8.675 19.2 7.37 18.053 7 16.5"
        fill="none"
      />
    </svg>
  );
}
