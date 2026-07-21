"use client";

import Image from "next/image";
import { PlusIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import type { BotPlatformInfo } from "@/app/api/__generated__/models/botPlatformInfo";

import { BotCardDmTile } from "./BotCardDmTile";
import { BotCardServerList } from "./BotCardServerList";
import { useBotCard } from "./useBotCard";

type Props = {
  platform: BotPlatformInfo;
};

export function BotCard({ platform }: Props) {
  const { isPending, unlinkServerLink, unlinkDmLink } = useBotCard();
  const serverLinks = platform.server_links ?? [];

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
        </div>
        {platform.add_bot_url ? (
          <Button
            as="NextLink"
            href={platform.add_bot_url}
            target="_blank"
            rel="noopener noreferrer"
            variant="primary"
            size="small"
            leftIcon={<PlusIcon size={16} />}
          >
            Add bot to {platform.display_name}
          </Button>
        ) : null}
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
          dmLink={platform.dm_link ?? null}
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
          Linked servers
        </Text>
        <BotCardServerList
          platformName={platform.display_name}
          serverLinks={serverLinks}
          isPending={isPending}
          onUnlink={unlinkServerLink}
        />
      </section>
    </Card>
  );
}
