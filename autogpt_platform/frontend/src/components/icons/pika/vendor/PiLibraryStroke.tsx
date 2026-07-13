import React from "react";

/**
 * PiLibraryStroke icon from the stroke style in general category.
 */
interface PiLibraryStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiLibraryStroke({
  size = 24,
  color,
  className,
  ariaLabel = "library icon",
  ...props
}: PiLibraryStrokeProps): JSX.Element {
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
        d="M6.4 9H4.6c-.56 0-.84 0-1.054.109a1 1 0 0 0-.437.437C3 9.76 3 10.04 3 10.6v8.8c0 .56 0 .84.109 1.054a1 1 0 0 0 .437.437C3.76 21 4.04 21 4.6 21h1.8c.56 0 .84 0 1.054-.109a1 1 0 0 0 .437-.437C8 20.24 8 19.96 8 19.4v-8.8c0-.56 0-.84-.109-1.054a1 1 0 0 0-.437-.437C7.24 9 6.96 9 6.4 9Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M11.4 3H9.6c-.56 0-.84 0-1.054.109a1 1 0 0 0-.437.437C8 3.76 8 4.04 8 4.6v14.8c0 .56 0 .84.109 1.054a1 1 0 0 0 .437.437C8.76 21 9.04 21 9.6 21h1.8c.56 0 .84 0 1.054-.109a1 1 0 0 0 .437-.437C13 20.24 13 19.96 13 19.4V4.6c0-.56 0-.84-.109-1.054a1 1 0 0 0-.437-.437C12.24 3 11.96 3 11.4 3Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="m17.005 6.376-1.71.566c-.53.176-.797.264-.966.435a1 1 0 0 0-.277.552c-.036.237.052.503.228 1.035l3.59 10.847c.176.532.264.797.435.966a1 1 0 0 0 .552.278c.238.036.503-.052 1.035-.228l1.709-.566c.532-.176.797-.264.966-.434a1 1 0 0 0 .278-.553c.036-.237-.052-.503-.228-1.034l-3.59-10.847c-.177-.532-.265-.798-.435-.967a1 1 0 0 0-.552-.277c-.238-.037-.504.051-1.035.227Z"
        fill="none"
      />
    </svg>
  );
}
