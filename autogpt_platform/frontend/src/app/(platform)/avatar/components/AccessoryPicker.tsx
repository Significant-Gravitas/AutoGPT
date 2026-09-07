import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  ACCESSORIES,
  type AccessoryId,
  type AvatarConfig,
} from "@/components/molecules/BotAvatar/helpers";
import { OptionTile } from "./OptionTile";
import { SectionHeading } from "./SectionHeading";

interface Props {
  config: AvatarConfig;
  onSelect: (accessory: AccessoryId) => void;
}

export function AccessoryPicker({ config, onSelect }: Props) {
  return (
    <section className="space-y-3">
      <SectionHeading title="Accessory" hint="one at most, the what-I-do cue" />
      <div
        role="radiogroup"
        aria-label="Accessory"
        className="grid grid-cols-4 gap-2 sm:grid-cols-8"
      >
        {ACCESSORIES.map((accessory) => (
          <OptionTile
            key={accessory.id}
            label={accessory.label}
            hint={accessory.hint}
            isSelected={config.accessory === accessory.id}
            onSelect={() => onSelect(accessory.id)}
          >
            <BotAvatar
              config={{ ...config, accessory: accessory.id }}
              size={56}
              animated={false}
              showBadge={false}
            />
          </OptionTile>
        ))}
      </div>
    </section>
  );
}
