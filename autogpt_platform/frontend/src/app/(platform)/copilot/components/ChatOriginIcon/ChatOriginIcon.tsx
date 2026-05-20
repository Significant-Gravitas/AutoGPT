"use client";

import Image from "next/image";
import { useState } from "react";

const PLATFORM_LOGOS: Record<string, { name: string; src: string }> = {
  discord: { name: "Discord", src: "/integrations/discord.png" },
  github: { name: "GitHub", src: "/integrations/github.png" },
  linear: { name: "Linear", src: "/integrations/linear.png" },
  slack: { name: "Slack", src: "/integrations/slack.png" },
  teams: { name: "Teams", src: "/integrations/teams.png" },
  telegram: { name: "Telegram", src: "/integrations/telegram.png" },
  whatsapp: { name: "WhatsApp", src: "/integrations/whatsapp.png" },
};

interface Props {
  sourcePlatform?: string | null;
}

export function ChatOriginIcon({ sourcePlatform }: Props) {
  const platform = sourcePlatform?.trim().toLocaleLowerCase();
  const logo = platform ? PLATFORM_LOGOS[platform] : undefined;
  const [brokenSrc, setBrokenSrc] = useState<string | null>(null);

  if (!logo || brokenSrc === logo.src) return null;

  return (
    <span
      className="inline-flex size-4 shrink-0 items-center justify-center"
      title={`From ${logo.name}`}
    >
      <Image
        src={logo.src}
        alt={logo.name}
        width={16}
        height={16}
        loading="lazy"
        className="size-4 object-contain"
        onError={() => setBrokenSrc(logo.src)}
      />
    </span>
  );
}
