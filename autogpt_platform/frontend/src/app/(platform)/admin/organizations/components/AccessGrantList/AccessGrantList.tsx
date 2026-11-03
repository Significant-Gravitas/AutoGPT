"use client";

import type { OrgAccessGrantInfo } from "@/app/api/__generated__/models/orgAccessGrantInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { formatGrantDate, PAGE_SIZE, pageCount } from "../../helpers";
import { DisableAccessDialog } from "../DisableAccessDialog/DisableAccessDialog";

interface Props {
  grants: OrgAccessGrantInfo[];
  total: number;
  isLoading: boolean;
  isError: boolean;
  onRetry: () => void;
  search: string;
  onSearchChange: (value: string) => void;
  page: number;
  onPageChange: (page: number) => void;
  isDisabling: boolean;
  onDisable: (userId: string, email: string) => Promise<boolean>;
}

export function AccessGrantList(props: Props) {
  const pages = pageCount(props.total, PAGE_SIZE);

  return (
    <div className="flex flex-col gap-4">
      <SearchInput
        value={props.search}
        onChange={props.onSearchChange}
        placeholder="Filter by email or name…"
        aria-label="Filter accounts"
        size="small"
      />
      <GrantRows {...props} />
      {pages > 1 ? (
        <div className="flex items-center justify-between gap-3">
          <Text variant="small" tone="muted">
            Page {props.page} of {pages}
          </Text>
          <div className="flex gap-2">
            <Button
              variant="secondary"
              size="small"
              disabled={props.page <= 1}
              onClick={() => props.onPageChange(props.page - 1)}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              size="small"
              disabled={props.page >= pages}
              onClick={() => props.onPageChange(props.page + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function GrantRows({
  grants,
  isLoading,
  isError,
  onRetry,
  search,
  isDisabling,
  onDisable,
}: Props) {
  if (isLoading) {
    return (
      <div className="flex flex-col gap-2" aria-busy="true">
        <Skeleton className="h-12 w-full rounded-lg" />
        <Skeleton className="h-12 w-full rounded-lg" />
      </div>
    );
  }
  if (isError) {
    return (
      <div className="flex items-center justify-between gap-4">
        <Text variant="small" tone="muted">
          Couldn&apos;t load the accounts.
        </Text>
        <Button variant="secondary" size="small" onClick={onRetry}>
          Retry
        </Button>
      </div>
    );
  }
  if (grants.length === 0) {
    return (
      <Text variant="small" tone="muted">
        {search.trim()
          ? "No account with organizations matches that."
          : "No account has organizations yet."}
      </Text>
    );
  }
  return (
    <ul className="divide-y divide-border-base">
      {grants.map((grant) => (
        <li
          key={grant.user_id}
          data-testid="org-access-row"
          className="flex flex-col gap-2 py-3 first:pt-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex min-w-0 flex-col">
            <Text variant="body-medium" as="span" className="truncate">
              {grant.email}
            </Text>
            <Text variant="small" as="span" tone="muted">
              {[
                grant.name,
                `On since ${formatGrantDate(grant.granted_at)}`,
                grant.granted_by_email ? `by ${grant.granted_by_email}` : null,
                grant.note,
              ]
                .filter(Boolean)
                .join(" · ")}
            </Text>
          </div>
          <DisableAccessDialog
            email={grant.email}
            isDisabling={isDisabling}
            onConfirm={() => onDisable(grant.user_id, grant.email)}
          />
        </li>
      ))}
    </ul>
  );
}
