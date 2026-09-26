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
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 rounded-md bg-zinc-50 p-3 text-sm">
      <dt className="text-zinc-500">Merchant</dt>
      <dd className="text-zinc-900">{merchant}</dd>
      <dt className="text-zinc-500">Paying on</dt>
      <dd className="break-all font-medium text-zinc-900">
        {host || "Unknown site"}
      </dd>
      <dt className="text-zinc-500">Total</dt>
      <dd className="font-medium text-zinc-900">
        {testMode ? `${total} · test, no charge` : total}
      </dd>
      <dt className="text-zinc-500">For</dt>
      <dd className="whitespace-pre-line text-zinc-700">{context}</dd>
    </dl>
  );
}
