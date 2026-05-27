import { TrashIcon, UsersIcon, WarningCircleIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { PlatformLinkInfo } from "@/app/api/__generated__/models/platformLinkInfo";

type Props = {
  platformName: string;
  serverLinks: PlatformLinkInfo[];
  isPending: (linkId: string) => boolean;
  onUnlink: (linkId: string) => void;
};

export function BotCardServerList({
  platformName,
  serverLinks,
  isPending,
  onUnlink,
}: Props) {
  if (serverLinks.length === 0) {
    return (
      <div className="rounded-large border border-dashed border-zinc-200 px-4 py-3">
        <Text variant="small" as="span" className="text-zinc-500">
          No servers linked yet. Use &quot;Add bot to {platformName}&quot; to
          invite the bot.
        </Text>
      </div>
    );
  }

  return (
    <ul className="flex flex-col gap-2">
      {serverLinks.map((link) => (
        <BotCardServerRow
          key={link.id}
          link={link}
          platformName={platformName}
          isPending={isPending(link.id)}
          onUnlink={() => onUnlink(link.id)}
        />
      ))}
    </ul>
  );
}

type RowProps = {
  link: PlatformLinkInfo;
  platformName: string;
  isPending: boolean;
  onUnlink: () => void;
};

function BotCardServerRow({
  link,
  platformName,
  isPending,
  onUnlink,
}: RowProps) {
  const botMissing = !link.server_name;
  const displayLabel = link.server_name ?? link.platform_server_id;

  return (
    <li className="flex items-center justify-between gap-3 rounded-large border border-zinc-200 px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        <UsersIcon size={20} className="shrink-0 text-zinc-500" />
        <div className="flex min-w-0 flex-col">
          <Text
            variant="body-medium"
            as="span"
            className="truncate text-textBlack"
          >
            {displayLabel}
          </Text>
          {botMissing ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="inline-flex items-center gap-1 text-xs text-amber-600">
                  <WarningCircleIcon size={14} weight="fill" /> Bot not in
                  server
                </span>
              </TooltipTrigger>
              <TooltipContent>
                {platformName} can&apos;t see this server, so we can&apos;t
                resolve its name. Re-invite the bot, or unlink to remove this
                row.
              </TooltipContent>
            </Tooltip>
          ) : (
            <Text variant="small" as="span" className="text-zinc-500">
              Linked by you
            </Text>
          )}
        </div>
      </div>
      <Button
        variant="outline"
        size="small"
        leftIcon={<TrashIcon size={16} />}
        loading={isPending}
        onClick={onUnlink}
        aria-label={`Unlink ${displayLabel}`}
      >
        Unlink
      </Button>
    </li>
  );
}
