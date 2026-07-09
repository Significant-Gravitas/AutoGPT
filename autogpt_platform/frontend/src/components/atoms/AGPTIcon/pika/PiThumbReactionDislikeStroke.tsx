import React from "react";

/**
 * PiThumbReactionDislikeStroke icon from the stroke style in general category.
 */
interface PiThumbReactionDislikeStrokeProps
  extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiThumbReactionDislikeStroke({
  size = 24,
  color,
  className,
  ariaLabel = "thumb-reaction-dislike icon",
  ...props
}: PiThumbReactionDislikeStrokeProps): JSX.Element {
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
        d="M16 11.064V12.5a2.5 2.5 0 0 0 5 0v-6a2.5 2.5 0 0 0-5 0v1.768m0 2.796c-.002.813-.015 1.285-.1 1.742a6 6 0 0 1-.436 1.389c-.226.5-.546.959-1.185 1.878l-3.012 4.326c-.077.11-.115.164-.15.205a1 1 0 0 1-1.382.123 3 3 0 0 1-.184-.176c-.164-.164-.245-.245-.315-.326a3 3 0 0 1-.602-2.845c.032-.102.073-.21.157-.426l.048-.124c.133-.346.2-.52.226-.656a1.5 1.5 0 0 0-1.199-1.748c-.137-.026-.322-.026-.694-.026H4.786c-.96 0-1.44 0-1.785-.195a1.5 1.5 0 0 1-.66-.766c-.142-.37-.07-.844.071-1.794l.532-3.555c.217-1.451.325-2.177.679-2.722A3 3 0 0 1 4.899 4.27C5.49 4 6.224 4 7.69 4h3.51c1.68 0 2.52 0 3.162.327a3 3 0 0 1 1.311 1.311c.29.57.323 1.297.327 2.63m0 2.796V8.268"
        fill="none"
      />
    </svg>
  );
}
