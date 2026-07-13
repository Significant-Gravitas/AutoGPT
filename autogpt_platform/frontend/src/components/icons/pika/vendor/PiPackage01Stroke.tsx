import React from "react";

/**
 * PiPackage01Stroke icon from the stroke style in development category.
 */
interface PiPackage01StrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiPackage01Stroke({
  size = 24,
  color,
  className,
  ariaLabel = "package-01 icon",
  ...props
}: PiPackage01StrokeProps): JSX.Element {
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
        d="M12 12v9.5m0-9.5L3.34 7.388M12 12l4.33-2.306M12 21.5q.326 0 .648-.065c.483-.099.938-.35 1.846-.853l4.012-2.22c.909-.503 1.363-.755 1.693-1.106.293-.312.513-.678.648-1.077.153-.45.153-.953.153-1.959V9.78c0-1.006 0-1.509-.153-1.96a3 3 0 0 0-.187-.432M12 21.5a3.3 3.3 0 0 1-.648-.065c-.483-.099-.937-.35-1.846-.853l-4.012-2.22c-.908-.503-1.363-.755-1.693-1.106a3 3 0 0 1-.648-1.077C3 15.73 3 15.226 3 14.22V9.78c0-1.006 0-1.509.153-1.96q.075-.223.187-.432m17.32 0a3 3 0 0 0-.46-.643c-.331-.352-.785-.604-1.694-1.107l-4.012-2.22c-.909-.503-1.363-.754-1.846-.853a3.25 3.25 0 0 0-1.296 0c-.483.099-.937.35-1.846.853L7.5 4.528m13.16 2.86-4.33 2.306M3.34 7.388q.186-.35.46-.643c.331-.352.786-.604 1.694-1.107L7.5 4.528m0 0 8.83 5.166"
        fill="none"
      />
    </svg>
  );
}
