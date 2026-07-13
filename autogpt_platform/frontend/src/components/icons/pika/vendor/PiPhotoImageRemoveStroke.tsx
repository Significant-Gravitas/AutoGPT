import React from "react";

/**
 * PiPhotoImageRemoveStroke icon from the stroke style in media category.
 */
interface PiPhotoImageRemoveStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiPhotoImageRemoveStroke({
  size = 24,
  color,
  className,
  ariaLabel = "photo-image-remove icon",
  ...props
}: PiPhotoImageRemoveStrokeProps): JSX.Element {
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
        d="M16 18h6m0-4v-3l-.001-1m-8.643 11H10c-.756 0-1.41 0-1.983-.01M22 10h-1c-1.393 0-2.09 0-2.676.06A11.5 11.5 0 0 0 8.06 20.324c-.02.2-.034.415-.043.665M22 10c-.008-2.15-.068-3.336-.544-4.27a5 5 0 0 0-2.185-2.185C18.2 3 16.8 3 14 3h-4c-2.8 0-4.2 0-5.27.545A5 5 0 0 0 2.545 5.73C2 6.8 2 8.2 2 11v2c0 2.8 0 4.2.545 5.27a5 5 0 0 0 2.185 2.185c.78.398 1.738.505 3.287.534M7.5 9.5a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"
        fill="none"
      />
    </svg>
  );
}
