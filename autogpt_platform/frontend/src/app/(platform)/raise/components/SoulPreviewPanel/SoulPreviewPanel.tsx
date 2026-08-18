"use client";

import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import { DEFAULT_GLASS_PARAMS } from "@/components/molecules/GlassOrb/GlassSurface";
import Image from "next/image";
import { kitBudgetLabel, kitToolsLabel, type RaiseKit } from "../../helpers";
import { roleLabelFor } from "../RoleStep/helpers";
import { SoulDetailCard } from "./SoulDetailCard";

const REVEAL =
  "duration-700 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

type Props = {
  name: string;
  role: string | null;
  avatarUrl: string | null;
  color: string | null;
  about: string | null;
  voiceLabel: string | null;
  kit: RaiseKit | null;
};

// Starts as just the orb — an expert with nothing to say about itself yet.
// Answers stack beneath it, which pushes the identity card up as they land.
export function SoulPreviewPanel({
  name,
  role,
  avatarUrl,
  color,
  about,
  voiceLabel,
  kit,
}: Props) {
  const roleLabel = roleLabelFor(role);
  const details = [
    { label: "About", value: about },
    { label: "Voice", value: voiceLabel },
    { label: "Weekly budget", value: kitBudgetLabel(kit) },
    { label: "Tools", value: kitToolsLabel(kit) },
  ].filter((detail) => Boolean(detail.value));

  return (
    <div className="flex w-full max-w-[19rem] flex-col items-center gap-3">
      <aside className="flex w-full flex-col items-center gap-8 rounded-[2rem] border border-border bg-background px-8 py-12 shadow-2xl">
        <span className="size-28 shrink-0 overflow-hidden rounded-full">
          {avatarUrl ? (
            <Image
              src={avatarUrl}
              alt={`${name || "Your expert"}'s picture`}
              width={112}
              height={112}
              className="size-28 rounded-full object-cover"
              unoptimized
            />
          ) : (
            <GlassOrb params={DEFAULT_GLASS_PARAMS} showRim={false} />
          )}
        </span>

        <div className="flex flex-col items-center gap-1.5">
          {name ? (
            <h2
              className={`text-center text-2xl font-semibold tracking-[-0.02em] text-foreground ${REVEAL}`}
            >
              {name}
            </h2>
          ) : (
            <p className="text-center text-2xl font-semibold tracking-[-0.02em] text-muted-foreground/50">
              No name yet
            </p>
          )}
          {roleLabel ? (
            <p
              className={`text-sm uppercase tracking-[0.12em] text-muted-foreground ${REVEAL}`}
            >
              {roleLabel}
            </p>
          ) : null}
        </div>
      </aside>

      {details.map((detail) => (
        <SoulDetailCard
          key={detail.label}
          label={detail.label}
          value={detail.value as string}
          color={color}
        />
      ))}
    </div>
  );
}
