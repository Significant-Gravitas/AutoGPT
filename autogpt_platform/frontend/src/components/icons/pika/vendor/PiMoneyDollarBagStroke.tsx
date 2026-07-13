import React from "react";

/**
 * PiMoneyDollarBagStroke icon from the stroke style in money-&-payments category.
 */
interface PiMoneyDollarBagStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiMoneyDollarBagStroke({
  size = 24,
  color,
  className,
  ariaLabel = "money-dollar-bag icon",
  ...props
}: PiMoneyDollarBagStrokeProps): JSX.Element {
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
        d="M12 10.303v.83m0 0h-.767c-.848 0-1.536.742-1.536 1.658s.688 1.658 1.536 1.658h1.535c.848 0 1.536.742 1.536 1.658s-.688 1.659-1.536 1.659H12m0-6.634h.816c.528 0 .994.288 1.27.726M12 17.765v.83m0-.83h-.816c-.527 0-.993-.287-1.27-.725m4.973-10.052a5.9 5.9 0 0 0-1.85-.298h-2.073q-.483 0-.945.076m4.868.222c6.185 2.031 8.735 13.808.922 14.777-2.53.313-5.088.313-7.617 0C.008 20.75 3.194 7.875 10.019 6.766m4.868.222 1.777-4.444-.65-.26a3.98 3.98 0 0 0-3.68.384 3.97 3.97 0 0 1-2.984.59l-.977-.195 1.646 3.703"
        fill="none"
      />
    </svg>
  );
}
