"use client";

import Image from "next/image";
import { useState } from "react";
import {
  getCategoryIcon,
  getProviderIconPath,
  RobotIcon,
  PlugIcon,
} from "./helpers";

interface Props {
  integrations: Array<{ name: string; type: "provider" | "category" }>;
  size?: "sm" | "md" | "lg";
  className?: string;
}

const SIZE_CONFIG = {
  sm: { icon: 20, gap: 8, lineWidth: 12 },
  md: { icon: 28, gap: 12, lineWidth: 16 },
  lg: { icon: 36, gap: 16, lineWidth: 20 },
} as const;

function ProviderIcon({ name, iconSize }: { name: string; iconSize: number }) {
  const [hasError, setHasError] = useState(false);

  if (hasError) {
    return <PlugIcon size={iconSize} className="text-zinc-400" />;
  }

  return (
    <Image
      src={getProviderIconPath(name)}
      alt={name}
      width={iconSize}
      height={iconSize}
      className="rounded-sm object-contain"
      onError={() => setHasError(true)}
    />
  );
}

function ConnectingLine({ width, height }: { width: number; height: number }) {
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="flex-shrink-0"
    >
      <line
        x1={0}
        y1={height / 2}
        x2={width}
        y2={height / 2}
        stroke="currentColor"
        strokeWidth={1.5}
        className="text-zinc-300"
      />
      <polygon
        points={`${width - 4},${height / 2 - 3} ${width},${height / 2} ${width - 4},${height / 2 + 3}`}
        fill="currentColor"
        className="text-zinc-300"
      />
    </svg>
  );
}

export function IntegrationLinkImage({
  integrations,
  size = "sm",
  className = "",
}: Props) {
  const config = SIZE_CONFIG[size];
  const items = integrations.slice(0, 3);

  if (items.length === 0) {
    return (
      <div
        className={`flex items-center justify-center rounded-small bg-zinc-50 ${className}`}
      >
        <RobotIcon size={config.icon} className="text-zinc-400" />
      </div>
    );
  }

  return (
    <div
      className={`flex items-center justify-center gap-0 rounded-small bg-zinc-50 ${className}`}
    >
      {items.map((item, i) => (
        <div key={`${item.name}-${i}`} className="flex items-center">
          {i > 0 && (
            <ConnectingLine width={config.lineWidth} height={config.icon} />
          )}
          <div className="flex items-center justify-center">
            {item.type === "provider" ? (
              <ProviderIcon name={item.name} iconSize={config.icon} />
            ) : (
              (() => {
                const CategoryIcon = getCategoryIcon(item.name);
                return (
                  <CategoryIcon size={config.icon} className="text-zinc-500" />
                );
              })()
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
