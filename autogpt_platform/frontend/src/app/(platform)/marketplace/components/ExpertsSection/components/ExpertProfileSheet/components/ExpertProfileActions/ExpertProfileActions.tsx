import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { AsyncStatus } from "@/types/async-status";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";

type Props = {
  expertName: string;
  isHired: boolean;
  isHiring: boolean;
  onHire: () => void;
  hiredExpertId: string | null;
  hiredLookup: AsyncStatus;
  onRetryHiredLookup: () => void;
};

export function ExpertProfileActions({
  expertName,
  isHired,
  isHiring,
  onHire,
  hiredExpertId,
  hiredLookup,
  onRetryHiredLookup,
}: Props) {
  return (
    <div className="relative mt-6">
      {isHired ? (
        <div className="flex flex-col gap-3">
          <div className="flex h-12 w-full items-center justify-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 text-base font-medium text-emerald-700">
            <Icon icon={CheckmarkCircle02Icon} size={20} />
            On your team
          </div>
          {hiredExpertId ? (
            <Button
              as="NextLink"
              href={`/copilot?expertId=${encodeURIComponent(hiredExpertId)}`}
              variant="primary"
              className="h-12 w-full rounded-full text-base"
            >
              Open chat
            </Button>
          ) : null}
        </div>
      ) : hiredLookup === "error" ? (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-3">
          <span className="text-sm text-zinc-600">
            Team status unavailable right now.
          </span>
          <Button variant="secondary" size="small" onClick={onRetryHiredLookup}>
            Retry
          </Button>
        </div>
      ) : (
        <Button
          variant="primary"
          onClick={onHire}
          loading={isHiring}
          disabled={hiredLookup === "loading"}
          className="h-12 w-full rounded-full text-base"
        >
          {`Hire ${expertName}`}
        </Button>
      )}
    </div>
  );
}
