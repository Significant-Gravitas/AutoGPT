import React from "react";

/**
 * PiRocketShipStroke icon from the stroke style in general category.
 */
interface PiRocketShipStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiRocketShipStroke({
  size = 24,
  color,
  className,
  ariaLabel = "rocket-ship icon",
  ...props
}: PiRocketShipStrokeProps): JSX.Element {
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
        d="m5.83 18.169-2.827 2.828m4.95-.707-.708.707m-3.535-4.95-.708.708M11.36 6.148H6.98c-.327 0-.543.06-.823.228L4.344 7.464c-.433.26-.65.39-.712.55a.5.5 0 0 0 .03.424c.082.15.314.249.779.448l3.473 1.488m3.445-4.226c-.641.642-1.247 1.416-2.17 2.594l-.863 1.104-.412.528m3.445-4.226c.294-.295.596-.561.94-.826.747-.577 2.313-1.415 3.208-1.717 1.306-.44 1.886-.484 3.048-.572.998-.076 1.817-.022 2.125.287.308.308.363 1.126.287 2.125-.088 1.161-.132 1.742-.572 3.048-.302.895-1.14 2.461-1.717 3.208a10 10 0 0 1-.826.94m-9.938-2.267c-.327.429-.518.709-.634.995a3 3 0 0 0 .157 2.586c.204.367.543.707 1.222 1.385.68.68 1.019 1.019 1.386 1.223a3 3 0 0 0 2.586.157c.286-.116.566-.307.994-.635m0 0 1.489 3.474c.199.464.298.697.448.78a.5.5 0 0 0 .424.028c.16-.062.29-.278.55-.711l1.088-1.814c.168-.28.228-.496.228-.823V12.64m-4.227 3.444q.23-.176.529-.411l1.104-.864c1.178-.922 1.952-1.528 2.594-2.17"
        fill="none"
      />
    </svg>
  );
}
