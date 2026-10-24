import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { AIRTABLE_HOME_URL } from "../helpers";

export function LoadingBases() {
  return (
    <div
      role="status"
      aria-label="Loading your Airtable bases"
      className="flex flex-col gap-2"
    >
      {[0, 1, 2].map((row) => (
        <Skeleton key={row} className="h-11 w-full rounded-2xl" />
      ))}
    </div>
  );
}

interface ErrorProps {
  message: string;
  onRetry: () => void;
  retrying: boolean;
}

export function BasesError({ message, onRetry, retrying }: ErrorProps) {
  return (
    <div
      role="alert"
      className="flex flex-col items-start gap-2 rounded-2xl bg-red-50 p-3"
    >
      <Text variant="small" className="text-red-700">
        {message}
      </Text>
      <Button
        type="button"
        size="small"
        variant="secondary"
        loading={retrying}
        onClick={onRetry}
      >
        Try again
      </Button>
    </div>
  );
}

interface RefreshProps {
  onRefresh: () => void;
  refreshing: boolean;
}

export function NoBasesYet({ onRefresh, refreshing }: RefreshProps) {
  return (
    <div className="flex flex-col items-start gap-2 rounded-2xl bg-zinc-50 p-3">
      <Text variant="small" className="text-zinc-700">
        Your Airtable has no bases yet. Create an empty one in Airtable, then
        come back and refresh this list.
      </Text>
      <NewBaseActions onRefresh={onRefresh} refreshing={refreshing} />
    </div>
  );
}

export function NewBaseActions({ onRefresh, refreshing }: RefreshProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Link href={AIRTABLE_HOME_URL} isExternal variant="secondary">
        Open Airtable
        <span className="sr-only"> (opens in a new tab)</span>
      </Link>
      <Button
        type="button"
        size="small"
        variant="ghost"
        loading={refreshing}
        onClick={onRefresh}
      >
        Refresh list
      </Button>
    </div>
  );
}
