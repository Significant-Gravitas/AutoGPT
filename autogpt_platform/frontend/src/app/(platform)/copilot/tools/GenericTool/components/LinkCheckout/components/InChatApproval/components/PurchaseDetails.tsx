import { SharedPageNote } from "../../SharedPageNote";

interface Props {
  merchant: string;
  address: string;
  sharedPage: boolean;
  total: string;
  context: string;
  testMode: boolean;
  assistantName: string;
}

export function PurchaseDetails({
  merchant,
  address,
  sharedPage,
  total,
  context,
  testMode,
  assistantName,
}: Props) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-surface-muted p-3 text-body">
      <dt className="text-foreground-tertiary">Merchant</dt>
      <dd className="text-foreground-strong">
        {merchant}
        <span className="text-foreground-tertiary">
          {` · named by ${assistantName}`}
        </span>
      </dd>
      <dt className="text-foreground-tertiary">Paying on</dt>
      <dd className="flex flex-col">
        <span className="font-medium break-all text-foreground-strong">
          {address || "Unknown site"}
        </span>
        {sharedPage && <SharedPageNote />}
      </dd>
      <dt className="text-foreground-tertiary">Total</dt>
      <dd className="font-medium text-foreground-strong">
        {testMode ? `${total} · test, no charge` : total}
      </dd>
      <dt className="text-foreground-tertiary">For</dt>
      <dd className="text-foreground-body">{context}</dd>
    </dl>
  );
}
