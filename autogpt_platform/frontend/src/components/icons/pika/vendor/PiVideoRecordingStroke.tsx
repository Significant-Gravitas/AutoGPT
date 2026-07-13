import React from "react";

/**
 * PiVideoRecordingStroke icon from the stroke style in media category.
 */
interface PiVideoRecordingStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiVideoRecordingStroke({
  size = 24,
  color,
  className,
  ariaLabel = "video-recording icon",
  ...props
}: PiVideoRecordingStrokeProps): JSX.Element {
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
        d="M21 15.604c-.027 1.387-.124 2.245-.481 2.946a4.5 4.5 0 0 1-1.969 1.969c-.7.357-1.56.454-2.946.481M21 8.396c-.027-1.387-.124-2.245-.481-2.946a4.5 4.5 0 0 0-1.969-1.97c-.7-.357-1.56-.454-2.946-.481M8.396 3c-1.387.027-2.245.124-2.946.481A4.5 4.5 0 0 0 3.48 5.45c-.357.7-.454 1.56-.481 2.946m0 7.208c.027 1.387.124 2.245.481 2.946a4.5 4.5 0 0 0 1.97 1.97c.7.357 1.56.454 2.946.481m6.645-7.288.901.72a.901.901 0 0 0 1.464-.703v-3.458a.901.901 0 0 0-1.464-.704l-.9.72a.9.9 0 0 0-.339.704v2.018a.9.9 0 0 0 .338.703Zm-4.843 1.892h.901c1.261 0 1.892 0 2.374-.245.424-.216.768-.561.984-.985.246-.482.246-1.113.246-2.374s0-1.892-.245-2.374a2.25 2.25 0 0 0-.985-.984c-.482-.246-1.113-.246-2.374-.246h-.901c-1.261 0-1.892 0-2.374.246a2.24 2.24 0 0 0-.984.984c-.246.482-.246 1.113-.246 2.374s0 1.892.246 2.374c.215.424.56.768.984.984.482.246 1.113.246 2.374.246Z"
        fill="none"
      />
    </svg>
  );
}
