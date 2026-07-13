import React from "react";

/**
 * PiStarStroke icon from the stroke style in general category.
 */
interface PiStarStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiStarStroke({
  size = 24,
  color,
  className,
  ariaLabel = "star icon",
  ...props
}: PiStarStrokeProps): JSX.Element {
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
        d="M9.15 6.247c.841-1.764 1.262-2.646 1.812-2.967a2.06 2.06 0 0 1 2.077 0c.549.32.97 1.203 1.812 2.967.25.523.374.785.553.993.21.244.473.436.77.56.254.105.54.143 1.115.219 1.939.255 2.908.383 3.382.807.554.495.8 1.249.642 1.975-.135.621-.844 1.294-2.262 2.64-.42.4-.63.599-.773.833a2.1 2.1 0 0 0-.294.906c-.022.273.03.558.136 1.128.356 1.922.534 2.884.277 3.465a2.06 2.06 0 0 1-1.68 1.221c-.633.064-1.492-.402-3.21-1.335-.51-.276-.764-.414-1.03-.478a2.06 2.06 0 0 0-.953 0c-.267.064-.522.202-1.03.478-1.72.933-2.578 1.4-3.21 1.335a2.06 2.06 0 0 1-1.681-1.22c-.257-.582-.079-1.544.277-3.466.106-.57.159-.855.136-1.128a2.06 2.06 0 0 0-.294-.906c-.143-.234-.353-.434-.773-.832-1.418-1.347-2.127-2.02-2.262-2.641a2.06 2.06 0 0 1 .642-1.975c.474-.424 1.444-.552 3.382-.807.574-.076.862-.114 1.115-.22.297-.123.56-.315.77-.56.179-.207.304-.469.553-.992Z"
        fill="none"
      />
    </svg>
  );
}
