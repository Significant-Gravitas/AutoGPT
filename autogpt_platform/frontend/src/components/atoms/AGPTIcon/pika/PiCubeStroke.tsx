import React from "react";

/**
 * PiCubeStroke icon from the stroke style in development category.
 */
interface PiCubeStrokeProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  color?: string;
  className?: string;
  ariaLabel?: string;
}

export default function PiCubeStroke({
  size = 24,
  color,
  className,
  ariaLabel = "cube icon",
  ...props
}: PiCubeStrokeProps): JSX.Element {
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
        d="M12 12v9.5m0-9.5L3.34 7.388M12 12l8.66-4.612M12 21.5q.326 0 .648-.065c.483-.099.938-.35 1.846-.853l4.012-2.22c.909-.503 1.363-.755 1.693-1.106.293-.312.513-.678.648-1.077.153-.45.153-.953.153-1.959V9.78c0-1.006 0-1.509-.153-1.96a3 3 0 0 0-.187-.432M12 21.5a3.3 3.3 0 0 1-.648-.065c-.483-.099-.937-.35-1.846-.853l-4.012-2.22c-.908-.503-1.363-.755-1.693-1.106a3 3 0 0 1-.648-1.077C3 15.73 3 15.226 3 14.22V9.78c0-1.006 0-1.509.153-1.96q.075-.223.187-.432m17.32 0a3 3 0 0 0-.46-.643c-.331-.352-.785-.604-1.694-1.107l-4.012-2.22c-.909-.503-1.363-.754-1.846-.853a3.25 3.25 0 0 0-1.296 0c-.483.099-.937.35-1.846.853l-4.012 2.22c-.908.503-1.363.755-1.693 1.107a3 3 0 0 0-.461.643"
        fill="none"
      />
    </svg>
  );
}
