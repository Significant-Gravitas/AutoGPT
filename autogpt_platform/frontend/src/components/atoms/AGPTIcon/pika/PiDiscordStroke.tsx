import React from "react";

/**
 * PiDiscordStroke icon from the stroke style in apps-&-social category.
 */
interface PiDiscordStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiDiscordStroke({
  size = 24,
  color,
  className,
  ariaLabel = "discord icon",
  ...props
}: PiDiscordStrokeProps): JSX.Element {
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
        d="M11 6h2.693a.5.5 0 0 0 .448-.278l.678-1.368a.476.476 0 0 1 .557-.253c.656.188 2.03.642 3.128 1.399 3.864 2.897 3.504 9.39 3.475 10.76a.5.5 0 0 1-.065.24C19.931 20 16.491 20 16.491 20l-1.166-2.426M13 6h-2.688a.5.5 0 0 1-.448-.277l-.683-1.37a.476.476 0 0 0-.556-.252c-.655.188-2.031.641-3.13 1.399-3.863 2.897-3.503 9.39-3.474 10.76a.5.5 0 0 0 .065.24C4.069 20 7.509 20 7.509 20l1.17-2.427M7 17c.6.225 1.155.416 1.678.573M17.004 17c-.6.225-1.156.416-1.68.574m-6.645 0c2.444.735 4.202.735 6.646 0M10.002 12a1 1 0 1 1-2 0 1 1 0 0 1 2 0Zm6.002 0a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
        fill="none"
      />
    </svg>
  );
}
