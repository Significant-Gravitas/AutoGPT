"use client";

import { MagnifyingGlassIcon, PlugIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";

import type { ConnectableProvider } from "../helpers";
import { ProviderRow } from "./ProviderRow";

interface Props {
  query: string;
  setQuery: (next: string) => void;
  providers: ConnectableProvider[];
  onSelect: (id: string) => void;
}

export function ListView({ query, setQuery, providers, onSelect }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-[#505057]">
        Pick a service to connect an API key or authorize with OAuth.
      </Text>

      <div className="relative w-full">
        <MagnifyingGlassIcon
          size={20}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[#83838C]"
        />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search services..."
          aria-label="Search services"
          className="h-[46px] w-full rounded-3xl border border-[#DADADC] bg-white pl-12 pr-4 text-sm leading-[22px] text-[#1F1F20] placeholder:text-[#83838C] focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400"
        />
      </div>

      {providers.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-[#DADADC] py-10 text-center">
          <PlugIcon size={24} className="text-[#83838C]" />
          <Text variant="body" className="text-[#505057]">
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
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white to-transparent"
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
