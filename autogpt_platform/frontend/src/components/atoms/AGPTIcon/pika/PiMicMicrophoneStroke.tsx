import React from "react";

/**
 * PiMicMicrophoneStroke icon from the stroke style in media category.
 */
interface PiMicMicrophoneStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMicMicrophoneStroke({
  size = 24,
  color,
  className,
  ariaLabel = "mic-microphone icon",
  ...props
}: PiMicMicrophoneStrokeProps): JSX.Element {
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
        d="M16.655 11.561a4.28 4.28 0 1 0-4.216-4.218m.002-.022 4.238 4.238-9.908 8.865a2.263 2.263 0 0 1-3.195-3.195z"
        fill="none"
      />
    </svg>
  );
}
