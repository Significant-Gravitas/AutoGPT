"use client";

import Image from "next/image";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import type { BotPlatformInfo } from "@/app/api/__generated__/models/botPlatformInfo";

import { BotCardDmTile } from "./BotCardDmTile";
import { BotCardServerList } from "./BotCardServerList";
import { useBotCard } from "./useBotCard";
import { ArrowUpRight01Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  platform: BotPlatformInfo;
};

export function BotCard({ platform }: Props) {
  const { isPending, unlinkServerLink, unlinkDmLink } = useBotCard();
  const serverLinks = platform.server_links ?? [];
  const pendingInstall = platform.pending_install ?? null;
  // Optional in the generated client (it has a server-side default).
  const serverNoun = platform.server_noun ?? "server";

  return (
    <Card className="flex flex-col gap-5 p-5">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Image
            src={`/integrations/${platform.icon}`}
            alt={`${platform.display_name} icon`}
            width={32}
            height={32}
            className="rounded-md"
          />
          <Text variant="large-medium" as="h2" className="text-textBlack">
            {platform.display_name}
          </Text>
          {pendingInstall ? (
            <Badge variant="info" size="small">
              Pending — finish in {platform.display_name}
            </Badge>
          ) : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {pendingInstall ? (
            <Button
              as="NextLink"
              href={pendingInstall.open_bot_url}
              target="_blank"
              rel="noopener noreferrer"
              variant="primary"
              size="small"
              rightIcon={<Icon icon={ArrowUpRight01Icon} size={16} />}
            >
              Open AutoGPT in {platform.display_name}
            </Button>
          ) : null}
          {/* Kept alongside the pending action, not replaced by it: adding the
              bot to a second {serverNoun} is a normal thing to want while the
              first install is still waiting on its DM. */}
          {platform.add_bot_url ? (
            // Same tab on purpose: the install round-trip ends by returning here,
            // so a new tab would just strand the user on a dead page.
            <Button
              as="NextLink"
              href={platform.add_bot_url}
              variant={pendingInstall ? "outline" : "primary"}
              size="small"
              leftIcon={<Icon icon={PlusSignIcon} size={16} />}
            >
              Add bot to {platform.display_name}
            </Button>
          ) : null}
        </div>
      </header>

      <section className="flex flex-col gap-2">
        <Text
          variant="small-medium"
          as="span"
          className="uppercase tracking-wide text-zinc-500"
        >
          Direct messages
        </Text>
        <BotCardDmTile
          platformName={platform.display_name}
          serverNoun={serverNoun}
          dmLink={platform.dm_link ?? null}
          pendingServerName={pendingInstall?.server_name ?? null}
          isPending={isPending}
          onUnlink={unlinkDmLink}
        />
      </section>

      <section className="flex flex-col gap-2">
        <Text
          variant="small-medium"
          as="span"
          className="uppercase tracking-wide text-zinc-500"
        >
          Linked {serverNoun}s
        </Text>
        <BotCardServerList
          platformName={platform.display_name}
          serverNoun={serverNoun}
          serverLinks={serverLinks}
          isPending={isPending}
          onUnlink={unlinkServerLink}
        />
      </section>
    </Card>
  );
}
