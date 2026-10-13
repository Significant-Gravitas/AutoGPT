interface Props {
  title: string;
  description: string;
  unit: string;
}

export function ChartHeading({ title, description, unit }: Props) {
  return (
    <div className="mb-5 flex items-start justify-between gap-3">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-zinc-800">{title}</h3>
        <p className="mt-1 text-xs text-zinc-500">{description}</p>
      </div>
      <span className="max-w-[35%] break-words rounded-md bg-purple-50 px-2 py-1 text-xs text-purple-700">
        {unit}
      </span>
    </div>
  );
}
