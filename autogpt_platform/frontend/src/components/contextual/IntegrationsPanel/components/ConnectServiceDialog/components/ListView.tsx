"use client";
import { useId } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";

import type { ConnectableProvider } from "../helpers";
import { PopularProviderTile } from "./PopularProviderTile";
import { ProviderRow } from "./ProviderRow";
import { Plug01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  query: string;
  setQuery: (next: string) => void;
  providers: ConnectableProvider[];
  popular: ConnectableProvider[];
  onSelect: (id: string) => void;
  description: string;
}

export function ListView({
  query,
  setQuery,
  providers,
  popular,
  onSelect,
  description,
}: Props) {
  const popularHeadingId = useId();
  const moreHeadingId = useId();
  const showSections = popular.length > 0;
  const isEmpty = providers.length === 0 && popular.length === 0;

  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" tone="body">
        {description}
      </Text>

      <div className="relative w-full">
        <Icon
          icon={Search01Icon}
          size={20}
          className="pointer-events-none absolute top-1/2 left-4 -translate-y-1/2 text-foreground-tertiary"
        />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search services..."
          aria-label="Search services"
          className="h-[46px] w-full rounded-3xl border border-border-base bg-surface pr-4 pl-12 text-body text-foreground-base placeholder:text-placeholder focus:border-brand focus:ring-1 focus:ring-brand focus:outline-none"
        />
      </div>

      {isEmpty ? (
        <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-border-base py-10 text-center">
          <Icon
            icon={Plug01Icon}
            size={24}
            className="text-foreground-tertiary"
          />
          <Text variant="body" tone="body">
            {query.trim()
              ? `No services match "${query.trim()}"`
              : "No services available"}
          </Text>
        </div>
      ) : (
        <div className="relative">
          {/* Radix wraps the list in a `display: table` div that grows to fit
              each row's full description, pushing cards past the edge. */}
          <ScrollArea className="h-[380px] pr-2 [&_[data-radix-scroll-area-viewport]>div]:!block">
            {showSections ? (
              <div className="flex flex-col gap-5 pb-4">
                <section
                  aria-labelledby={popularHeadingId}
                  className="flex flex-col gap-2.5"
                >
                  <Text
                    variant="eyebrow"
                    as="h3"
                    tone="secondary"
                    id={popularHeadingId}
                  >
                    Popular
                  </Text>
                  <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {popular.map((provider) => (
                      <li key={provider.id}>
                        <PopularProviderTile
                          provider={provider}
                          onSelect={onSelect}
                        />
                      </li>
                    ))}
                  </ul>
                </section>
                {providers.length > 0 && (
                  <section
                    aria-labelledby={moreHeadingId}
                    className="flex flex-col gap-2.5"
                  >
                    <Text
                      variant="eyebrow"
                      as="h3"
                      tone="secondary"
                      id={moreHeadingId}
                    >
                      More services
                    </Text>
                    <ProviderList providers={providers} onSelect={onSelect} />
                  </section>
                )}
              </div>
            ) : (
              <div className="pb-4">
                <ProviderList providers={providers} onSelect={onSelect} />
              </div>
            )}
          </ScrollArea>
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-surface to-transparent"
          />
        </div>
      )}
    </div>
  );
}

function ProviderList({
  providers,
  onSelect,
}: Pick<Props, "providers" | "onSelect">) {
  return (
    <ul className="flex flex-col gap-2">
      {providers.map((provider) => (
        <li key={provider.id}>
          <ProviderRow provider={provider} onSelect={onSelect} />
        </li>
      ))}
    </ul>
  );
}

export function ListLoading() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-5 w-3/4" />
      <Skeleton className="h-[46px] w-full rounded-3xl" />
      <Skeleton className="h-4 w-16" />
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-16 w-full rounded-xl" />
        ))}
      </div>
      <Skeleton className="h-4 w-24" />
      <div className="flex flex-col gap-2">
        {[0, 1].map((i) => (
          <Skeleton key={i} className="h-20 w-full rounded-xl" />
        ))}
      </div>
    </div>
  );
}
