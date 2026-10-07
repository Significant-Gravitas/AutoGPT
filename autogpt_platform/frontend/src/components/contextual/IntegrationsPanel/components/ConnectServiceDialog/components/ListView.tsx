"use client";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ScrollArea } from "@/components/atoms/ScrollArea/ScrollArea";

import type { ConnectableProvider } from "../helpers";
import { ProviderRow } from "./ProviderRow";
import { Plug01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  query: string;
  setQuery: (next: string) => void;
  providers: ConnectableProvider[];
  onSelect: (id: string) => void;
  description: string;
}

export function ListView({
  query,
  setQuery,
  providers,
  onSelect,
  description,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" tone="secondary">
        {description}
      </Text>

      <div className="relative w-full">
        <Icon
          icon={Search01Icon}
          size={20}
          className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-muted-foreground"
        />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search services..."
          aria-label="Search services"
          className="h-[46px] w-full rounded-3xl border border-zinc-200 bg-white pr-4 pl-12 text-sm leading-[22px] text-black placeholder:text-zinc-500 focus:border-purple-400 focus:ring-1 focus:ring-purple-400 focus:outline-hidden"
        />
      </div>

      {providers.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-zinc-200 py-10 text-center">
          <Icon icon={Plug01Icon} size={24} className="text-muted-foreground" />
          <Text variant="body" tone="secondary" unmask={false}>
            {query.trim()
              ? `No services match "${query.trim()}"`
              : "No services available"}
          </Text>
        </div>
      ) : (
        <div className="relative">
          <ScrollArea className="h-[380px] pr-2">
            <ul className="flex flex-col gap-2 pb-4">
              {providers.map((provider) => (
                <li key={provider.id}>
                  <ProviderRow provider={provider} onSelect={onSelect} />
                </li>
              ))}
            </ul>
          </ScrollArea>
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-linear-to-t from-white to-transparent"
          />
        </div>
      )}
    </div>
  );
}

export function ListLoading() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-5 w-3/4" />
      <Skeleton className="h-[46px] w-full rounded-3xl" />
      <div className="flex flex-col gap-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-16 w-full rounded-xl" />
        ))}
      </div>
    </div>
  );
}
