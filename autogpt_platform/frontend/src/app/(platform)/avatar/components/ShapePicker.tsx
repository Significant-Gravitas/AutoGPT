import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  SHAPES,
  type AvatarConfig,
  type ShapeId,
} from "@/components/molecules/BotAvatar/helpers";
import { OptionTile } from "./OptionTile";
import { SectionHeading } from "./SectionHeading";

interface Props {
  config: AvatarConfig;
  outline: boolean;
  onSelect: (shape: ShapeId) => void;
}

export function ShapePicker({ config, outline, onSelect }: Props) {
  return (
    <section className="space-y-3">
      <SectionHeading title="Shape" hint="silhouette identity" />
      <div
        role="radiogroup"
        aria-label="Shape"
        className="grid grid-cols-3 gap-2 sm:grid-cols-6"
      >
        {SHAPES.map((shape) => (
          <OptionTile
            key={shape.id}
            label={shape.label}
            hint={shape.hint}
            isSelected={config.shape === shape.id}
            onSelect={() => onSelect(shape.id)}
          >
            <BotAvatar
              config={{ ...config, shape: shape.id, accessory: "none" }}
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
