interface Props {
  merchant: string;
  host: string;
  total: string;
  context: string;
  testMode: boolean;
}

export function PurchaseDetails({
  merchant,
  host,
  total,
  context,
  testMode,
}: Props) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-surface-muted p-3 text-body">
      <dt className="text-foreground-tertiary">Merchant</dt>
      <dd className="text-foreground-strong">{merchant}</dd>
      <dt className="text-foreground-tertiary">Paying on</dt>
      <dd className="font-medium break-all text-foreground-strong">
        {host || "Unknown site"}
      </dd>
      <dt className="text-foreground-tertiary">Total</dt>
      <dd className="font-medium text-foreground-strong">
        {testMode ? `${total} · test, no charge` : total}
      </dd>
      <dt className="text-foreground-tertiary">For</dt>
      <dd className="whitespace-pre-line text-foreground-body">{context}</dd>
    </dl>
  );
}
