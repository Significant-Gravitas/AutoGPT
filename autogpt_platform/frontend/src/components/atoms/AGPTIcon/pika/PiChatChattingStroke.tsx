import React from "react";

/**
 * PiChatChattingStroke icon from the stroke style in communication category.
 */
interface PiChatChattingStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiChatChattingStroke({
  size = 24,
  color,
  className,
  ariaLabel = "chat-chatting icon",
  ...props
}: PiChatChattingStrokeProps): JSX.Element {
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
        d="m17.05 18.718 1.74.124c.762.055 1.143.082 1.428-.053a1.2 1.2 0 0 0 .57-.57c.136-.286.109-.667.054-1.429l-.124-1.74c-.02-.28-.03-.42-.032-.562-.004-.258-.007-.122.009-.38.009-.142.088-.811.248-2.15Q21 11.486 21 11A8 8 0 0 0 6.385 6.5m7.415 9.1a5.4 5.4 0 1 0-10.733.856c.096.6.144.9.152.992.012.139.011.107.01.247 0 .093-.008.204-.023.427l-.1 1.386c-.036.515-.055.772.037.964a.8.8 0 0 0 .385.385c.192.091.45.073.964.036l1.385-.099c.224-.015.335-.024.428-.024.14 0 .108-.001.247.011.093.008.393.056.992.152Q7.963 21 8.4 21a5.39 5.39 0 0 0 4.203-2.01A5.38 5.38 0 0 0 13.8 15.6Z"
        fill="none"
      />
    </svg>
  );
}
