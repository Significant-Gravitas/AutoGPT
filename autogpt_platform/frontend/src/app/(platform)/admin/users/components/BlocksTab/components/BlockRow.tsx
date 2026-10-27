"use client";

import type { AccessBlock } from "@/app/api/__generated__/models/accessBlock";
import { Icon } from "@/components/atoms/Icon/Icon";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { GlobeIcon, MailAtSign01Icon } from "@hugeicons/core-free-icons";
import {
  formatBlockedAddress,
  formatDateTime,
  formatEnds,
} from "../../../helpers";
import { UnblockDialog } from "../../AccessDialogs/UnblockDialog";

const HOUR_MS = 60 * 60 * 1000;

interface Props {
  block: AccessBlock;
}

export function BlockRow({ block }: Props) {
  const isIP = block.kind === "IP";
  const endsSoon =
    block.expires_at !== null &&
    block.expires_at.getTime() - Date.now() < HOUR_MS;

  return (
    <li className="grid grid-cols-1 gap-x-4 gap-y-1 py-2.5 first:pt-0 last:pb-0 lg:grid-cols-[minmax(0,12rem)_minmax(0,1fr)_minmax(0,9rem)_8rem_minmax(0,10rem)_auto] lg:items-center">
      <div className="flex min-w-0 items-center gap-2">
        <Icon
          icon={isIP ? GlobeIcon : MailAtSign01Icon}
          size={16}
          className="shrink-0 text-foreground-tertiary"
        />
        <Text
          variant={isIP ? "body" : "body-medium"}
          as="span"
          className={cn(
            "truncate",
            isIP && "font-mono text-body-sm leading-inherit",
          )}
          translate="no"
        >
          {isIP ? formatBlockedAddress(block.value) : block.value}
        </Text>
      </div>
      <OverflowText
        value={block.reason}
        variant="body"
        className="text-foreground-body"
      />
      <Text
        variant="small"
        as="span"
        className="truncate text-foreground-tertiary"
      >
        {block.created_by_email ?? "unknown"}
      </Text>
      <Text variant="small" as="span" className="text-foreground-tertiary">
        {formatDateTime(block.created_at)}
      </Text>
      <Text
        variant="small"
        as="span"
        className={
          !block.expires_at
            ? "text-foreground-tertiary"
            : endsSoon
              ? "text-warning-foreground"
              : "text-foreground-body"
        }
      >
        {formatEnds(block.expires_at)}
      </Text>
      <div className="lg:text-right">
        <UnblockDialog block={block} triggerVariant="ghost" />
      </div>
    </li>
  );
}
