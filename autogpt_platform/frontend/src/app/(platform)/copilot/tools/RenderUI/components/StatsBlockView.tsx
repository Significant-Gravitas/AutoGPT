import type { StatsBlock } from "@/app/api/__generated__/models/statsBlock";
import { Text } from "@/components/atoms/Text/Text";
import { TEXT_TONE } from "../helpers";
import { BlockCard } from "./BlockCard";

interface Props {
  block: StatsBlock;
}

export function StatsBlockView({ block }: Props) {
  return (
    <BlockCard>
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {block.items.map((item, index) => (
          <div key={index} className="min-w-0">
            <dt>
              <Text variant="small" tone="secondary" className="break-words">
                {item.label}
              </Text>
            </dt>
            <dd>
              <Text variant="h3" tabular className="break-words">
                {item.value}
              </Text>
              {item.change ? (
                <Text
                  variant="small"
                  weight="medium"
                  tone={TEXT_TONE[item.tone ?? "neutral"]}
                >
                  {item.change}
                </Text>
              ) : null}
            </dd>
          </div>
        ))}
      </dl>
    </BlockCard>
  );
}
