import React from "react";

/**
 * PiFolderOpenStroke icon from the stroke style in files-&-folders category.
 */
interface PiFolderOpenStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFolderOpenStroke({
  size = 24,
  color,
  className,
  ariaLabel = "folder-open icon",
  ...props
}: PiFolderOpenStrokeProps): JSX.Element {
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
        d="M20.961 9.997c-.289-.017-.647-.017-1.102-.017H8.445c-.875 0-1.313 0-1.68.151a2 2 0 0 0-.82.627c-.242.314-.357.736-.587 1.581L4.1 16.96c-.374 1.374-.561 2.061-.402 2.604a2 2 0 0 0 .963 1.192m16.3-10.758c.39.023.654.078.861.205.317.194.55.5.655.856.119.407-.021.922-.302 1.952l-.89 3.271c-.46 1.69-.69 2.535-1.175 3.163a4 4 0 0 1-1.64 1.253c-.733.303-1.61.303-3.36.303H7.187c-1.338 0-2.047 0-2.527-.245m16.3-10.758c-.047-.783-.155-1.339-.397-1.813a4 4 0 0 0-1.748-1.748C17.96 6 16.84 6 14.6 6h-1.316c-.47 0-.704 0-.917-.065a1.5 1.5 0 0 1-.517-.276c-.172-.142-.302-.337-.562-.728l-.575-.862c-.261-.391-.391-.586-.563-.728a1.5 1.5 0 0 0-.517-.276C9.42 3 9.185 3 8.716 3H8.4c-2.24 0-3.36 0-4.216.436a4 4 0 0 0-1.748 1.748C2 6.04 2 7.16 2 9.4v5.2c0 2.24 0 3.36.436 4.216a4 4 0 0 0 1.748 1.748q.224.114.477.19"
        fill="none"
      />
    </svg>
  );
}
