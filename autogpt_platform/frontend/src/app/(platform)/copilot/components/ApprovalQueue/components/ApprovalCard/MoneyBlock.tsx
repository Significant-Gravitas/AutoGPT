import type { ApprovalSpend } from "../../helpers";

interface Props {
  spend: ApprovalSpend;
}

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
});

function dollars(microdollars: number) {
  return usd.format(microdollars / 1_000_000);
}

export function MoneyBlock({ spend }: Props) {
  const share =
    spend.ceiling > 0 ? Math.min(1, spend.spent / spend.ceiling) : 1;
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-zinc-200 px-3 py-2.5 text-sm tabular-nums">
      <div className="flex justify-between gap-3 text-zinc-500">
        <span>This step</span>
        <span className="text-zinc-900">about {dollars(spend.estimate)}</span>
      </div>
      <div className="flex justify-between gap-3 text-zinc-500">
        <span>Spent in this chat</span>
        <span className="text-zinc-900">
          {dollars(spend.spent)} of {dollars(spend.ceiling)}
        </span>
      </div>
      <div
        role="progressbar"
        aria-label="Spent of this chat's ceiling"
        aria-valuenow={Math.round(share * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
        className="h-1 overflow-hidden rounded-full bg-zinc-100"
      >
        <div
          className="h-full rounded-full bg-amber-500"
          style={{ width: `${share * 100}%` }}
        />
      </div>
      <p className="text-xs text-zinc-500">
        Approving runs this step and adds {dollars(spend.unit)} to this
        chat&apos;s ceiling.
      </p>
    </div>
  );
}
