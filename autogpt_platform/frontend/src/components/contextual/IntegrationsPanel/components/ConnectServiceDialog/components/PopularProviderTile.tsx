"use client";

import type { ConnectableProvider } from "../helpers";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ProviderLogo } from "./ProviderLogo";

interface Props {
  provider: ConnectableProvider;
  onSelect: (id: string) => void;
}

export function PopularProviderTile({ provider, onSelect }: Props) {
  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      // TODO(tokens): no exact role for focus-visible:ring-purple-400 #925cf7
      className="group flex min-h-16 w-full items-center gap-3 rounded-xl border border-border-base bg-surface py-2.5 pl-3 pr-2.5 text-left transition-colors hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 active:bg-surface-hover"
    >
      <ProviderLogo provider={provider} size="sm" />
      <span className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          className="truncate leading-5.5 text-foreground-emphasis"
        >
          {provider.name}
        </Text>
        {provider.description && (
          <Text
            variant="small"
            as="span"
            tone="secondary"
            className="truncate leading-4"
            title={provider.description}
          >
            {provider.description}
          </Text>
        )}
      </span>
      <Icon
        icon={PlusSignIcon}
        size={16}
        aria-hidden
        className="shrink-0 text-foreground-secondary transition-colors group-hover:text-foreground-emphasis"
      />
    </button>
  );
}
