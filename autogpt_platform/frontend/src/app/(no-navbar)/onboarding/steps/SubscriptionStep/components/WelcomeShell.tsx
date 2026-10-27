import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import type { ReactNode } from "react";
import { type CoverArt, OttoCover } from "./OttoCover";

interface Props {
  art: CoverArt;
  children: ReactNode;
}

export function WelcomeShell({ art, children }: Props) {
  return (
    <FadeIn className="w-full px-4 pt-12 lg:pt-0">
      <section
        aria-labelledby="welcome-title"
        className="mx-auto grid w-full max-w-[60rem] overflow-hidden rounded-[1.75rem] border border-border-base/80 bg-surface shadow-[0_1px_2px_rgba(24,24,27,0.04),0_32px_64px_-32px_rgba(24,24,27,0.18)] lg:grid-cols-2"
      >
        <OttoCover art={art} />
        <div className="flex flex-col gap-6 px-5 pb-6 pt-5 sm:gap-7 sm:px-10 sm:py-9">
          {children}
        </div>
      </section>
    </FadeIn>
  );
}
