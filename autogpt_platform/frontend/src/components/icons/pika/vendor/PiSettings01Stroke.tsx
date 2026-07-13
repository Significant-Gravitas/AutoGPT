import React from "react";

/**
 * PiSettings01Stroke icon from the stroke style in general category.
 */
interface PiSettings01StrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiSettings01Stroke({
  size = 24,
  color,
  className,
  ariaLabel = "settings-01 icon",
  ...props
}: PiSettings01StrokeProps): JSX.Element {
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
        d="M10.2 4.12c.632-.618.947-.927 1.31-1.043.319-.103.661-.103.98 0 .363.116.678.425 1.31 1.043l.303.297c.273.267.409.4.567.497q.211.128.453.187c.18.044.37.046.752.05l.425.005c.883.009 1.325.013 1.663.188.298.153.54.395.694.693.174.338.178.78.187 1.663l.005.425c.004.382.006.573.05.752q.059.242.187.453c.096.158.23.295.497.567l.297.304c.618.63.927.946 1.043 1.309.103.319.103.661 0 .98-.116.363-.425.678-1.043 1.31l-.297.303c-.267.273-.4.409-.497.567q-.128.211-.187.453c-.044.18-.046.37-.05.752l-.005.425c-.009.883-.013 1.325-.187 1.663-.154.298-.396.54-.694.694-.338.174-.78.178-1.663.187l-.425.005c-.382.004-.573.006-.752.05q-.24.059-.453.187c-.158.096-.294.23-.567.497l-.304.297c-.63.618-.946.927-1.309 1.043a1.6 1.6 0 0 1-.98 0c-.363-.116-.678-.425-1.31-1.043l-.303-.297c-.272-.267-.409-.4-.567-.497a1.6 1.6 0 0 0-.453-.187c-.18-.044-.37-.046-.752-.05l-.425-.005c-.883-.009-1.325-.013-1.663-.187a1.6 1.6 0 0 1-.693-.694c-.175-.338-.18-.78-.188-1.663l-.005-.425c-.004-.382-.006-.573-.05-.752a1.6 1.6 0 0 0-.187-.453c-.096-.158-.23-.294-.497-.567l-.297-.304c-.618-.63-.927-.946-1.043-1.309a1.6 1.6 0 0 1 0-.98c.116-.363.425-.678 1.043-1.31l.297-.303c.267-.272.4-.409.497-.567q.128-.212.187-.453c.044-.18.046-.37.05-.752l.005-.425c.009-.883.013-1.325.188-1.663a1.6 1.6 0 0 1 .693-.693c.338-.175.78-.18 1.663-.188l.425-.005c.382-.004.573-.006.752-.05q.242-.059.453-.187c.158-.096.295-.23.567-.497z"
        fill="none"
      />
      <path
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2"
        d="M10.991 12c0-.552.457-1 1.01-1s1.008.448 1.008 1-.457 1-1.009 1-1.009-.448-1.009-1Z"
        fill="none"
      />
    </svg>
  );
}
