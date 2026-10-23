import { cn } from "@/lib/utils";
import Image from "next/image";

// Otto's approved v1.1 master, unchanged, on the warm studio colour it is
// exported on; the trial version sets a clay envelope beside him.
const COVER_ART = {
  letter: {
    src: "/onboarding/otto-welcome.webp",
    alt: "Otto, your personal Head of AI, beside a sealed envelope",
    width: 1196,
    height: 782,
  },
  hello: {
    src: "/autogpt-characters/v1.1/otto/neutral-transparent/1024.png",
    alt: "Otto, your personal Head of AI",
    width: 1024,
    height: 1024,
  },
} as const;

export type CoverArt = keyof typeof COVER_ART;

interface Props {
  art: CoverArt;
}

export function OttoCover({ art }: Props) {
  const image = COVER_ART[art];
  return (
    <div className="flex items-center justify-center bg-[#FDFCF8] px-6 pb-2 pt-6 sm:py-10 lg:px-10">
      <Image
        src={image.src}
        alt={image.alt}
        width={image.width}
        height={image.height}
        priority
        sizes="(min-width: 1024px) 26rem, 14rem"
        className={cn(
          "h-auto select-none",
          art === "letter"
            ? "w-[13rem] sm:w-[20rem] lg:w-full lg:max-w-[26rem]"
            : "w-[10rem] sm:w-[16rem] lg:w-[85%] lg:max-w-[22rem]",
        )}
      />
    </div>
  );
}
