"use client";

import { useEffect } from "react";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { type IconSet, useIconSetStore } from "@/services/icon-set/useIconSet";

const ICON_SET_OPTIONS: {
  value: IconSet;
  label: string;
  description: string;
}[] = [
  {
    value: "pika",
    label: "Pika",
    description: "Rounded line icons (default)",
  },
  {
    value: "phosphor",
    label: "Phosphor",
    description: "Classic Phosphor icon set",
  },
];

export default function SettingsAppearancePage() {
  useEffect(() => {
    document.title = "Appearance – AutoGPT Platform";
  }, []);

  const iconSet = useIconSetStore((state) => state.iconSet);
  const setIconSet = useIconSetStore((state) => state.setIconSet);
  const flagPika = useGetFlag(Flag.PIKA_ICONS);
  const selected: IconSet = iconSet ?? (flagPika ? "pika" : "phosphor");

  return (
    <div className="flex flex-col gap-6 pb-8">
      <div className="flex flex-col gap-1">
        <Text variant="lead-medium">Appearance</Text>
        <Text variant="body" className="text-zinc-500">
          Customise how AutoGPT looks for you.
        </Text>
      </div>

      <section className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <div className="flex flex-col gap-0.5">
          <Text variant="body-medium">Icon set</Text>
          <Text variant="body" className="text-sm text-zinc-500">
            Choose the icon style used across the app.
          </Text>
        </div>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          {ICON_SET_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={selected === option.value}
              onClick={() => setIconSet(option.value)}
              className={cn(
                "flex flex-col items-start gap-0.5 rounded-2xl border p-3 text-left transition-colors [corner-shape:squircle]",
                selected === option.value
                  ? "border-zinc-800 bg-zinc-50"
                  : "border-zinc-200 hover:bg-zinc-50",
              )}
            >
              <span className="text-sm font-medium text-zinc-900">
                {option.label}
              </span>
              <span className="text-xs text-zinc-500">
                {option.description}
              </span>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
