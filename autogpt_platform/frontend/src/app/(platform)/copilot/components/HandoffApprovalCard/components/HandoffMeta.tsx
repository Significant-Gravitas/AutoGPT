interface Props {
  facts: { label: string; value: string | null }[];
}

/** Expected back · By · Cap · Owner, each only when it is known. */
export function HandoffMeta({ facts }: Props) {
  const known = facts.filter(
    (fact): fact is { label: string; value: string } => !!fact.value,
  );
  return (
    <dl className="grid grid-cols-2 gap-4 border-t border-zinc-100 pt-2.5 sm:flex">
      {known.map((fact) => (
        <div key={fact.label} className="flex min-w-0 flex-1 flex-col gap-0.5">
          <dt className="text-xs leading-[18px] text-zinc-500">{fact.label}</dt>
          <dd className="truncate text-sm font-medium leading-[22px] text-zinc-900">
            {fact.value}
          </dd>
        </div>
      ))}
    </dl>
  );
}
