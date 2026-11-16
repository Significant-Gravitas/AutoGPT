interface Props {
  label: string;
  value: string;
  subtitle?: string;
}

function SummaryCard({ label, value, subtitle }: Props) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-body text-foreground-tertiary">{label}</div>
      <div className="text-heading-md font-bold">{value}</div>
      {subtitle && (
        <div className="mt-1 text-caption text-foreground-tertiary">
          {subtitle}
        </div>
      )}
    </div>
  );
}

export { SummaryCard };
