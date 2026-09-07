import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  EXPRESSIONS,
  type ExpressionId,
} from "@/components/molecules/BotAvatar/expressions";
import type { AvatarConfig } from "@/components/molecules/BotAvatar/helpers";
import { OptionTile } from "./OptionTile";
import { SectionHeading } from "./SectionHeading";

interface Props {
  config: AvatarConfig;
  outline: boolean;
  selected: ExpressionId | null;
  onSelect: (expression: ExpressionId | null) => void;
}

export function ExpressionPicker({
  config,
  outline,
  selected,
  onSelect,
}: Props) {
  return (
    <section className="space-y-3">
      <SectionHeading
        title="Expression"
        hint="each status cycles its own pool; pin one to inspect it"
      />
      <div
        role="radiogroup"
        aria-label="Expression"
        className="grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-9"
      >
        <OptionTile
          label="Auto"
          hint="from status"
          isSelected={selected === null}
          onSelect={() => onSelect(null)}
        >
          <BotAvatar
            config={config}
            size={56}
            animated={false}
            outline={outline}
            showBadge={false}
          />
        </OptionTile>
        {EXPRESSIONS.map((expression) => (
          <OptionTile
            key={expression.id}
            label={expression.label}
            isSelected={selected === expression.id}
            onSelect={() => onSelect(expression.id)}
          >
            <BotAvatar
              config={config}
              expression={expression.id}
              size={56}
              animated={false}
              outline={outline}
              showBadge={false}
            />
          </OptionTile>
        ))}
      </div>
    </section>
  );
}
