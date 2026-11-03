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
      className="group flex min-h-16 w-full items-center gap-3 rounded-xl border border-border-base bg-surface py-2.5 pr-2.5 pl-3 text-left transition-colors hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-brand focus-visible:outline-none active:bg-surface-hover"
    >
      <ProviderLogo provider={provider} size="sm" />
      <span className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          tone="emphasis"
          className="truncate"
        >
          {provider.name}
        </Text>
        {provider.description && (
          <Text
            variant="small-tight"
            as="span"
            tone="secondary"
            className="truncate"
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
        tone="secondary"
        motion="colors"
        className="shrink-0 group-hover:text-foreground-emphasis"
      />
    </button>
  );
}
