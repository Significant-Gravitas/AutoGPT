import React from "react";

/**
 * PiCameraDefaultStroke icon from the stroke style in devices category.
 */
interface PiCameraDefaultStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiCameraDefaultStroke({
  size = 24,
  color,
  className,
  ariaLabel = "camera-default icon",
  ...props
}: PiCameraDefaultStrokeProps): JSX.Element {
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
        d="M16.757 6c1.157 0 1.735 0 2.202.158a3 3 0 0 1 1.884 1.883C21 8.508 21 9.086 21 10.243V15.2c0 1.68 0 2.52-.327 3.162a3 3 0 0 1-1.311 1.311C18.72 20 17.88 20 16.2 20H7.816c-1.68 0-2.52 0-3.162-.327a3 3 0 0 1-1.311-1.31c-.327-.643-.327-1.483-.327-3.163v-4.97c0-1.145 0-1.717.154-2.18a3 3 0 0 1 1.896-1.896C5.528 6 6.101 6 7.246 6c.213 0 .32 0 .419-.02a1 1 0 0 0 .504-.27c.072-.071.131-.16.25-.338l1.106-1.66c.174-.26.26-.39.375-.484a1 1 0 0 1 .345-.184C10.387 3 10.543 3 10.856 3h2.288c.313 0 .47 0 .611.044a1 1 0 0 1 .345.184c.114.094.201.224.375.485L15.58 5.37c.12.18.18.27.252.342a1 1 0 0 0 .5.267c.101.021.209.021.425.021Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M12 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M17 10v.01"
        fill="none"
      />
    </svg>
  );
}
