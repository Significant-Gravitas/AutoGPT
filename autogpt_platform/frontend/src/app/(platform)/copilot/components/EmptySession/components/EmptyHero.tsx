"use client";

import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { CSSProperties } from "react";
import type { RecipientOption } from "../../ChatInput/components/RecipientChip";
import { TextGenerateEffect } from "./TextGenerateEffect/TextGenerateEffect";
import { getAvatarGlow, getExpertIntroSuffix } from "../helpers";

interface Props {
  name: string | null;
  /** Null holds the line back (recipient still resolving) without moving
   * the composer below it. */
  intro: string | null;
  recipient?: RecipientOption;
  isExpert?: boolean;
  expertRole?: string | null;
}

// The regular empty-session hero: one line naming who to tell. While a
// greeting is being written GreetingLoader renders in its place instead.
export function EmptyHero({ intro, recipient, isExpert, expertRole }: Props) {
  if (intro === null) {
    return (
      <div aria-hidden className="mb-6 pt-8 text-lead leading-normal md:pt-0">
        &nbsp;
      </div>
    );
  }

  if (recipient) {
    return (
      <div className="mb-6 pt-8 text-lead leading-relaxed font-normal tracking-tight text-foreground-strong md:pt-0">
        {isExpert ? "I'm" : "Tell"}{" "}
        <span className="ml-1.5 inline-flex items-center gap-2 align-middle">
          <span
            className="inline-flex rounded-lg shadow-(--avatar-glow)"
            style={
              { "--avatar-glow": getAvatarGlow(recipient) } as CSSProperties
            }
          >
            {recipient.id === null && !recipient.avatarUrl ? (
              <AutopilotAvatar size={32} shape="soft" />
            ) : (
              <ExpertAvatar
                name={recipient.name}
                avatarUrl={recipient.avatarUrl}
                size={32}
                shape="soft"
              />
            )}
          </span>
          {recipient.name}
        </span>
        {isExpert
          ? getExpertIntroSuffix(expertRole ?? null)
          : " about your work, and it will find what to automate."}
      </div>
    );
  }

  // Keyed on the text so switching recipient re-types the line.
  return (
    <TextGenerateEffect
      key={intro}
      className="mb-6 pt-8 !font-normal md:pt-0 [&_div]:!text-lead [&_div]:!leading-relaxed [&_div]:!tracking-tight [&_div]:!text-foreground-strong [&>div]:!mt-0"
      duration={0.6}
      words={intro}
    />
  );
}
