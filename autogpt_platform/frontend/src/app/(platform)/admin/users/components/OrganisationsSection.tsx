"use client";

import { useGetAdminGetUserOrganizations } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AdminSection } from "../../components/AdminSection";
import { formatMonth } from "../helpers";

interface Props {
  userId: string;
}

export function OrganisationsSection({ userId }: Props) {
  const {
    data: organisations,
    isLoading,
    refetch,
  } = useGetAdminGetUserOrganizations(userId, { query: { select: okData } });

  return (
    <AdminSection
      title="Organisations"
      description="Every account has a personal organisation."
    >
      {isLoading ? (
        <div className="flex flex-col gap-2" aria-busy="true">
          <Skeleton className="h-10 w-full rounded-lg" />
          <Skeleton className="h-10 w-full rounded-lg" />
        </div>
      ) : !organisations ? (
        <div className="flex items-center justify-between gap-4">
          <Text variant="small" className="text-zinc-500">
            Couldn&apos;t load organisations.
          </Text>
          <Button
            variant="secondary"
            size="small"
            onClick={() => void refetch()}
          >
            Retry
          </Button>
        </div>
      ) : organisations.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          No organisations.
        </Text>
      ) : (
        <ul className="divide-y divide-zinc-200">
          {organisations.map((organisation) => (
            <li
              key={organisation.id}
              className="flex items-center justify-between gap-3 py-2.5 first:pt-0 last:pb-0"
            >
              <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5">
                <Text variant="body-medium" as="span" className="truncate">
                  {organisation.name}
                </Text>
                <Text
                  variant="small"
                  as="span"
                  className="font-mono text-zinc-500"
                  translate="no"
                >
                  {organisation.slug}
                </Text>
                {organisation.is_personal ? (
                  <Badge variant="info" size="small">
                    Personal
                  </Badge>
                ) : null}
              </div>
              <Text
                variant="small"
                as="span"
                className="shrink-0 text-zinc-500"
              >
                {formatMonth(new Date(organisation.created_at))}
              </Text>
            </li>
          ))}
        </ul>
      )}
    </AdminSection>
  );
}
