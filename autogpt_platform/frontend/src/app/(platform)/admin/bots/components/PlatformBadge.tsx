import Image from "next/image";

import { PLATFORM_OPTIONS } from "./helpers";

const PLATFORM_LABELS: Record<string, string> = Object.fromEntries(
  PLATFORM_OPTIONS.filter((option) => option.value !== "all").map((option) => [
    option.value.toUpperCase(),
    option.label,
  ]),
);

interface Props {
  platform: string;
}

export function PlatformBadge({ platform }: Props) {
  const key = platform.toUpperCase();
  const label = PLATFORM_LABELS[key] ?? platform;

  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap">
      <Image
        src={`/integrations/${key.toLowerCase()}.png`}
        alt={`${label} icon`}
        width={16}
        height={16}
        className="rounded-sm"
      />
      {label}
    </span>
  );
}
