import { Text } from "@/components/atoms/Text/Text";
import type { Turn } from "../helpers";

interface Props {
  turn: Turn;
  onChange: (turn: Turn) => void;
}

const AXES: { key: keyof Turn; label: string; limit: number }[] = [
  { key: "yaw", label: "Turn", limit: 70 },
  { key: "pitch", label: "Tilt", limit: 35 },
];

export function TurnControl({ turn, onChange }: Props) {
  return (
    <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
      {AXES.map((axis) => (
        <label key={axis.key} className="flex items-center gap-2">
          <Text variant="small" as="span" className="w-8 text-zinc-500">
            {axis.label}
          </Text>
          <input
            type="range"
            min={-axis.limit}
            max={axis.limit}
            step={1}
            value={turn[axis.key]}
            aria-label={axis.label}
            onChange={(event) =>
              onChange({ ...turn, [axis.key]: Number(event.target.value) })
            }
            className="h-1.5 w-36 cursor-pointer appearance-none rounded-full bg-zinc-200 accent-zinc-900"
          />
          <Text
            variant="small"
            as="span"
            className="w-9 tabular-nums text-zinc-500"
          >
            {turn[axis.key]}°
          </Text>
        </label>
      ))}
    </div>
  );
}
