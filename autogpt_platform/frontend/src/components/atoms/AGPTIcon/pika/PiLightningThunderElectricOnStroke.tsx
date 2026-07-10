import React from "react";

/**
 * PiLightningThunderElectricOnStroke icon from the stroke style in weather category.
 */
interface PiLightningThunderElectricOnStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiLightningThunderElectricOnStroke({
  size = 24,
  color,
  className,
  ariaLabel = "lightning-thunder-electric-on icon",
  ...props
}: PiLightningThunderElectricOnStrokeProps): JSX.Element {
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
        d="m6.054 8.38 5.756-4.513c1.638-1.284 2.457-1.926 3.018-1.863.485.055.912.366 1.136.828.26.533-.003 1.58-.53 3.673l-.44 1.753c-.211.838-.316 1.257-.244 1.62.064.32.221.611.45.83.259.249.652.361 1.438.585l.51.146c1.512.432 2.268.648 2.57 1.087.264.382.348.872.23 1.328-.137.526-.772 1.014-2.041 1.99l-5.652 4.342c-1.628 1.25-2.442 1.876-3 1.81a1.46 1.46 0 0 1-1.129-.831c-.257-.532.003-1.565.522-3.63l.394-1.568c.211-.838.316-1.257.244-1.621a1.6 1.6 0 0 0-.45-.83c-.259-.248-.652-.36-1.438-.585l-.568-.162c-1.496-.427-2.244-.64-2.546-1.077a1.63 1.63 0 0 1-.234-1.322c.132-.524.756-1.013 2.004-1.99Z"
        fill="none"
      />
    </svg>
  );
}
