import type { StatsBlock } from "@/app/api/__generated__/models/statsBlock";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { type BlockTone, TEXT_TONE } from "../helpers";

// A faint wash of the tone's colour from the top corner, so a good or bad
// number reads before it is read.
const TILE_TINT: Record<BlockTone, string> = {
  neutral: "from-surface-muted",
  positive: "from-success-surface",
  negative: "from-error-surface",
  warning: "from-warning-surface",
};

interface Props {
  block: StatsBlock;
}

export function StatsBlockView({ block }: Props) {
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {block.items.map((item, index) => {
        const tone = item.tone ?? "neutral";
        return (
          <div
            key={index}
            className={cn(
              "flex min-w-0 flex-col gap-1 rounded-2xl bg-gradient-to-br to-surface p-4 smooth-shadow-ring-sm",
              TILE_TINT[tone],
            )}
          >
            <dt>
              <Text
                variant="small-medium"
                tone="secondary"
                className="break-words"
              >
                {item.label}
              </Text>
            </dt>
            <dd className="flex flex-col gap-0.5">
              <Text variant="h3" tabular className="break-words">
                {item.value}
              </Text>
              {item.change ? (
                <Text variant="small-medium" tone={TEXT_TONE[tone]} tabular>
                  {item.change}
                </Text>
              ) : null}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
