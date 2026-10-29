"use client";

import type { ConnectableProvider } from "../helpers";
import { PlusSignIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Badge } from "@/components/atoms/Badge/Badge";
import { ProviderLogo } from "./ProviderLogo";

interface Props {
  provider: ConnectableProvider;
  onSelect: (id: string) => void;
}

export function ProviderRow({ provider, onSelect }: Props) {
  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      // TODO(tokens): no exact role for focus-visible:ring-purple-400 #925cf7
      className="group flex min-h-20 w-full items-center gap-3 rounded-xl border border-border-base bg-surface px-[0.875rem] py-[0.625rem] text-left transition-colors hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400 active:bg-surface-hover"
    >
      <ProviderLogo provider={provider} size="md" />
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-body font-medium leading-5.5 text-foreground-emphasis">
          <span>{provider.name}</span>
          {provider.mcpServer &&
            provider.mcpServer.connection_mode !== "hosted" && (
              <Badge variant="warning" size="small">
                Setup required
              </Badge>
            )}
        </span>
        <span className="truncate text-caption leading-5 text-foreground-tertiary">
          {provider.description ?? provider.id}
        </span>
      </span>
      <span
        aria-hidden
        // TODO(tokens): no exact role for bg-zinc-700 #505057; #505057 is foreground-body, a text colour
        className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-zinc-700 text-action-foreground transition-transform group-hover:bg-action group-active:scale-[0.96]"
      >
        <Icon icon={PlusSignIcon} size={18} />
      </span>
    </button>
  );
}
