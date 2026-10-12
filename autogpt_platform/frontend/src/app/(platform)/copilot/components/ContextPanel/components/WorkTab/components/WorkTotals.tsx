interface Props {
  totals: { label: string; value: string }[];
}

/** Working · Waiting · Spent (· Cap once caps exist). */
export function WorkTotals({ totals }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium uppercase leading-4 tracking-[0.06em] text-zinc-500">
        Totals
      </span>
      <div className="flex gap-2">
        {totals.map(({ label, value }) => (
          <div
            key={label}
            className="flex min-w-0 flex-1 flex-col gap-0.5 rounded-xl bg-zinc-50 px-3 py-2.5"
          >
            <span className="text-xs text-zinc-500">{label}</span>
            <span className="truncate font-poppins text-base font-medium text-zinc-900">
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
