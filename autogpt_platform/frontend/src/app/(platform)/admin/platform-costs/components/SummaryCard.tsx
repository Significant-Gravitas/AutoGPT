interface Props {
  label: string;
  value: string;
  subtitle?: string;
}

function SummaryCard({ label, value, subtitle }: Props) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-sm text-foreground-tertiary">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
      {subtitle && (
        <div className="mt-1 text-xs text-foreground-tertiary">{subtitle}</div>
      )}
    </div>
  );
}

export { SummaryCard };
