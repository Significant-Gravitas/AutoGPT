"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { CaretUpDownIcon } from "@phosphor-icons/react";
import Image from "next/image";
import type { Persona } from "../personas";

interface Props {
  persona: Persona;
  isOpen: boolean;
  onToggle: () => void;
}

// Matches the avatar badge motion in settings/profile: the badge arcs along the
// circle's edge from bottom-right to top-right. The outer span sweeps it, the
// inner span counter-rotates so the icon never appears to spin.
const ARC =
  "origin-center transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] motion-reduce:transition-none";

export function PersonaAvatar({ persona, isOpen, onToggle }: Props) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-haspopup="listbox"
      aria-expanded={isOpen}
      aria-label={`Persona: ${persona.name}, ${persona.role}. Change persona`}
      className="group relative z-10 mx-auto mb-6 block size-32 cursor-pointer rounded-full outline-none focus-visible:ring-2 focus-visible:ring-offset-2"
      style={{ ["--persona-accent" as string]: persona.accent }}
    >
      {/* While the dial is open the wheel's own bottom item takes this spot and
          scrolls with the ring, so the static copy fades out of the way. */}
      <span
        className={`flex size-32 items-center justify-center overflow-hidden rounded-full border transition-[colors,opacity] duration-300 ease-out ${isOpen ? "opacity-0" : "opacity-100"}`}
        style={{
          borderColor: `${persona.accent}55`,
          backgroundColor: persona.tint,
          boxShadow: `0 2px 8px rgba(0,0,0,0.04), 0 0 32px -4px ${persona.accent}59`,
        }}
      >
        {persona.image ? (
          <Image
            src={persona.image}
            alt=""
            width={128}
            height={128}
            className="size-full rounded-full object-cover"
            priority
          />
        ) : (
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-16" />
        )}
      </span>
      {!isOpen && (
        <span
          className={`${ARC} pointer-events-none absolute inset-0 group-hover:-rotate-90 motion-reduce:rotate-0`}
        >
          <span
            className={`${ARC} absolute bottom-1 right-1 flex size-9 items-center justify-center rounded-full border-2 border-white bg-white text-black shadow-[0_3px_10px_-2px_rgba(15,15,20,0.25)] group-hover:rotate-90 motion-reduce:rotate-0`}
          >
            <CaretUpDownIcon size={18} />
          </span>
        </span>
      )}
    </button>
  );
}
