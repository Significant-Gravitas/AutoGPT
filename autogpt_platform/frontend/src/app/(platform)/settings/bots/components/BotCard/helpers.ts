import type { BotPlatformInfo } from "@/app/api/__generated__/models/botPlatformInfo";

export function getAddBotLabel(platform: BotPlatformInfo, serverNoun: string) {
  // With a DM link the card leads with messaging the bot, so adding it to a
  // group reads as the optional extra it is.
  return platform.dm_url
    ? `Add to a ${serverNoun}`
    : `Add bot to ${platform.display_name}`;
}
