"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import Image from "next/image";
import { getProfessionImageSrc, Profession } from "../helpers";
import { SelectedExpert } from "../useExpertPreview";

interface Props {
  professions: Profession[];
  rowIndex: number;
  reverse?: boolean;
  durationSeconds: number;
  paused: boolean;
  onSelect: (selected: SelectedExpert) => void;
}

export function MarqueeRow({
  professions,
  rowIndex,
  reverse,
  durationSeconds,
  paused,
  onSelect,
}: Props) {
  const track = [...professions, ...professions];

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-40 bg-gradient-to-r from-white via-white/80 to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-40 bg-gradient-to-l from-white via-white/80 to-transparent" />
      <div
        className={cn(
          "flex w-max animate-marquee-x motion-reduce:animate-none",
          reverse && "[animation-direction:reverse]",
        )}
        style={{
          animationDuration: `${durationSeconds}s`,
          animationPlayState: paused ? "paused" : "running",
        }}
      >
        {track.map((profession, index) => {
          const layoutId = `${rowIndex}-${index}-${profession.slug}`;

          return (
            <button
              key={layoutId}
              type="button"
              aria-label={`View ${profession.label}`}
              onClick={() => onSelect({ profession, layoutId })}
              className="flex w-36 shrink-0 flex-col items-center gap-2 rounded-2xl px-4 py-2 outline-none ring-0 transition-transform duration-150 ease-out focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300 active:scale-[0.97]"
            >
              <motion.div layoutId={layoutId} className="h-24 w-24">
                <Image
                  src={getProfessionImageSrc(profession.slug)}
                  alt=""
                  width={112}
                  height={112}
                  className="h-full w-full object-contain"
                />
              </motion.div>
              <Text variant="small" className="text-center text-zinc-500">
                {profession.label}
              </Text>
            </button>
          );
        })}
      </div>
    </div>
  );
}
