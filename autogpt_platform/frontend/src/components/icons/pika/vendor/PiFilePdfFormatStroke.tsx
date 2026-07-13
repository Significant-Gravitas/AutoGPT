import React from "react";

/**
 * PiFilePdfFormatStroke icon from the stroke style in files-&-folders category.
 */
interface PiFilePdfFormatStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiFilePdfFormatStroke({
  size = 24,
  color,
  className,
  ariaLabel = "file-pdf-format icon",
  ...props
}: PiFilePdfFormatStrokeProps): JSX.Element {
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
        d="M14 2.058V3.2c0 1.68 0 2.52.327 3.162a3 3 0 0 0 1.311 1.311C16.28 8 17.12 8 18.8 8h1.142M14 2.058C13.607 2 13.136 2 12.349 2H10.4c-2.24 0-3.36 0-4.216.436a4 4 0 0 0-1.748 1.748C4 5.04 4 6.16 4 8.4V10m10-7.942q.143.02.277.053c.408.098.798.26 1.156.478.404.248.75.594 1.442 1.286l1.25 1.25c.692.692 1.038 1.038 1.286 1.442a4 4 0 0 1 .479 1.156q.031.134.052.277m0 0c.058.394.058.864.058 1.651V10M3 21v-7h1.5a2.5 2.5 0 0 1 0 5H3m14 2v-7h4m0 4h-4m-7-4v7h.5a3.5 3.5 0 1 0 0-7z"
        fill="none"
      />
    </svg>
  );
}
