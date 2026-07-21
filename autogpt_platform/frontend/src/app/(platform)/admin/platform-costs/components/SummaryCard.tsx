interface Props {
  label: string;
  value: string;
  subtitle?: string;
}

function SummaryCard({ label, value, subtitle }: Props) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
      {subtitle && (
        <div className="mt-1 text-xs text-muted-foreground">{subtitle}</div>
      )}
    </div>
  );
}

export { SummaryCard };
