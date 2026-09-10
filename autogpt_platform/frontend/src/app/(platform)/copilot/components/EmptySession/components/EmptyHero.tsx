"use client";

import { Text } from "@/components/atoms/Text/Text";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { EditNameDialog } from "./EditNameDialog/EditNameDialog";

interface Props {
  name: string;
  /** Null holds the line back (recipient still resolving) without moving
   * the composer below it. */
  intro: string | null;
}

// The regular empty-session hero. While a greeting is being written
// GreetingLoader renders in its place instead, and its orb travels into
// the intro card's heading under a shared layout id.
export function EmptyHero({ name, intro }: Props) {
  return (
    <>
      <div className="mb-1 flex items-center justify-center gap-3">
        <Text variant="h3" className="!text-[1.375rem] text-zinc-700">
          Hey, <span className="text-violet-600">{name}</span>
          <EditNameDialog currentName={name} />
        </Text>
      </div>
      {intro === null ? (
        <div aria-hidden className="mb-8 text-[1.375rem] leading-normal">
          &nbsp;
        </div>
      ) : (
        // Keyed on the text so switching recipient re-types the line.
        <TextGenerateEffect
          key={intro}
          className="mb-8 !font-normal [&>div]:!mt-0 [&_div]:!text-[1.375rem] [&_div]:!leading-normal [&_div]:!tracking-normal"
          duration={0.6}
          words={intro}
        />
      )}
    </>
  );
}
