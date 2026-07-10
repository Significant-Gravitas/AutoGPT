import React from "react";

/**
 * PiThumbReactionLikeStroke icon from the stroke style in general category.
 */
interface PiThumbReactionLikeStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiThumbReactionLikeStroke({
  size = 24,
  color,
  className,
  ariaLabel = "thumb-reaction-like icon",
  ...props
}: PiThumbReactionLikeStrokeProps): JSX.Element {
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
        d="m15.162 7.17.047-.123a7 7 0 0 0 .157-.426 3 3 0 0 0-.602-2.846 7 7 0 0 0-.315-.325 3 3 0 0 0-.184-.176 1 1 0 0 0-1.382.123c-.035.04-.073.095-.15.205L9.721 7.928c-.64.919-.959 1.378-1.185 1.878-.202.443-.348.91-.436 1.389-.1.54-.1 1.1-.1 2.218v1.788c0 1.68 0 2.52.327 3.162a3 3 0 0 0 1.311 1.31c.642.328 1.482.328 3.162.328h3.509c1.467 0 2.2 0 2.792-.27a3 3 0 0 0 1.276-1.098c.354-.546.462-1.271.68-2.722l.53-3.555c.143-.95.214-1.425.072-1.794a1.5 1.5 0 0 0-.66-.766c-.344-.195-.825-.195-1.785-.195h-2.386c-.372 0-.557 0-.694-.026a1.5 1.5 0 0 1-1.199-1.748c.026-.137.093-.31.226-.657Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M3 11.5a2.5 2.5 0 0 1 5 0v6a2.5 2.5 0 1 1-5 0z"
        fill="none"
      />
    </svg>
  );
}
