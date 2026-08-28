import { isOverAutocompact, windowPercent } from "../helpers";

interface Props {
  context: number;
}

export function MiniBar({ context }: Props) {
  return (
    <span className="relative h-1 w-10 overflow-hidden rounded-full bg-zinc-200">
      <span
        className={
          "absolute inset-y-0 left-0 rounded-full " +
          (isOverAutocompact(context) ? "bg-amber-400" : "bg-zinc-500")
        }
        style={{ width: `${windowPercent(context)}%` }}
      />
    </span>
  );
}
