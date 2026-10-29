export interface PlatformLogo {
  name: string;
  src: string;
}

export const PLATFORM_LOGOS: Record<string, PlatformLogo> = {
  discord: { name: "Discord", src: "/integrations/discord.png" },
  github: { name: "GitHub", src: "/integrations/github.png" },
  linear: { name: "Linear", src: "/integrations/linear.png" },
  slack: { name: "Slack", src: "/integrations/slack.png" },
  teams: { name: "Teams", src: "/integrations/teams.png" },
  telegram: { name: "Telegram", src: "/integrations/telegram.png" },
  whatsapp: { name: "WhatsApp", src: "/integrations/whatsapp.png" },
};

export function resolvePlatformLogo(
  sourcePlatform?: string | null,
): PlatformLogo | undefined {
  const key = sourcePlatform?.trim().toLowerCase();
  return key ? PLATFORM_LOGOS[key] : undefined;
}
