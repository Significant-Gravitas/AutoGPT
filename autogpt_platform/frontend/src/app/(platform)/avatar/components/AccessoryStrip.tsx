"use client";

import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  ACCESSORIES,
  type AvatarConfig,
} from "@/components/molecules/BotAvatar/helpers";

interface Props {
  config: AvatarConfig;
  onPick: (config: AvatarConfig) => void;
}

export function AccessoryStrip({ config, onPick }: Props) {
  return (
    <section className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-4">
      <Text variant="body-medium" as="h2">
        Every accessory on this face
      </Text>
      <div className="flex flex-wrap gap-3">
        {ACCESSORIES.map((accessory) => {
          const option = { ...config, accessory: accessory.id };
          return (
            <button
              key={accessory.id}
              type="button"
              onClick={() => onPick(option)}
              title={`${accessory.label} — ${accessory.hint}`}
              className="flex flex-col items-center gap-1 rounded-xl p-1 transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <div className="flex items-end gap-1">
                <BotAvatar
                  config={option}
                  size={64}
                  animated={false}
                  showBadge={false}
                  title={accessory.label}
                />
                <BotAvatar
                  config={option}
                  size={24}
                  animated={false}
                  showBadge={false}
                  title={`${accessory.label} small`}
                />
              </div>
              <span className="text-[11px] text-zinc-600">
                {accessory.label}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
