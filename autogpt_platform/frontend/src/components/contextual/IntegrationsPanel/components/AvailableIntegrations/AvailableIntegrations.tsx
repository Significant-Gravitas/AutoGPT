"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ProviderRow } from "../ConnectServiceDialog/components/ProviderRow";
import { useAvailableIntegrations } from "./useAvailableIntegrations";

interface Props {
  query: string;
  onSelect: (providerId: string) => void;
}

export function AvailableIntegrations({ query, onSelect }: Props) {
  const catalog = useAvailableIntegrations(query);

  return (
    <section
      aria-labelledby="available-integrations-heading"
      className="flex flex-col gap-4 pb-6 pt-6"
    >
      <div className="flex flex-col gap-1 px-4">
        <Text
          variant="small-medium"
          as="h2"
          id="available-integrations-heading"
          className="uppercase tracking-[0.06em] text-zinc-600"
        >
          Available integrations
        </Text>
        <Text variant="small" className="text-zinc-500">
          Connect your services, including official MCP integrations. Some
          services need additional setup.
        </Text>
      </div>
      {catalog.isLoading ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {[0, 1, 2, 3].map((id) => (
            <Skeleton key={id} className="h-20 rounded-xl" />
          ))}
        </div>
      ) : catalog.isError ? (
        <ErrorCard
          context="available integrations"
          responseError={
            catalog.error instanceof Error
              ? { message: catalog.error.message }
              : undefined
          }
          onRetry={() => catalog.refetch()}
        />
      ) : (
        <>
          <ul className="grid gap-3 sm:grid-cols-2">
            {catalog.providers.map((provider) => (
              <li key={provider.id} className="min-w-0">
                <ProviderRow provider={provider} onSelect={onSelect} />
              </li>
            ))}
          </ul>
          <Text variant="small" className="px-4 text-zinc-500" role="status">
            {catalog.total === 0
              ? "No available services match your search."
              : `${catalog.providers.length} of ${catalog.total} services`}
          </Text>
          {catalog.hasMore && (
            <Button variant="secondary" size="small" onClick={catalog.showMore}>
              Show more services
            </Button>
          )}
        </>
      )}
    </section>
  );
}
