import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  UserQuestion01Icon,
  UserSearch01Icon,
} from "@hugeicons/core-free-icons";
import { getSpendingSearchHref } from "../helpers";

export function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-2 py-16 text-center">
      <Icon
        icon={UserSearch01Icon}
        size={32}
        className="text-foreground-subtle"
        aria-hidden
      />
      <Text variant="large-medium" as="h2">
        Look up a user
      </Text>
      <Text variant="body" className="text-foreground-secondary">
        Paste a user ID or the complete email address.
      </Text>
    </div>
  );
}

export function NotFoundState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center gap-2 py-16 text-center">
      <Icon
        icon={UserQuestion01Icon}
        size={32}
        className="text-foreground-subtle"
        aria-hidden
      />
      <Text variant="large-medium" as="h2" className="break-all">
        No user matches <span className="font-medium">{query}</span>
      </Text>
      <Text variant="body" className="text-foreground-secondary">
        Lookup is exact: a full user ID or the complete email address.
      </Text>
      <Link
        variant="secondary"
        href={getSpendingSearchHref(query)}
        className="mt-2"
      >
        Search spending history for &ldquo;{query}&rdquo; instead
      </Link>
    </div>
  );
}

export function LookupSkeleton() {
  return (
    <div
      className="flex flex-col gap-6"
      aria-busy="true"
      aria-live="polite"
      aria-label="Loading user"
    >
      <Skeleton className="h-[88px] w-full rounded-[18px]" />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex flex-col gap-6">
          <Skeleton className="h-[220px] w-full rounded-[18px]" />
          <Skeleton className="h-[160px] w-full rounded-[18px]" />
          <Skeleton className="h-[200px] w-full rounded-[18px]" />
          <Skeleton className="h-[180px] w-full rounded-[18px]" />
        </div>
        <div className="flex flex-col gap-6">
          <Skeleton className="h-[260px] w-full rounded-[18px]" />
          <Skeleton className="h-[112px] w-full rounded-[18px]" />
        </div>
      </div>
    </div>
  );
}
