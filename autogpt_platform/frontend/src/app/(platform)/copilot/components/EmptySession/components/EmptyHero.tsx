"use client";

import { Text } from "@/components/atoms/Text/Text";
import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { cn } from "@/lib/utils";
import {
  ORB_PURPLE,
  SMALL_ORB_PARAMS,
} from "../../OnboardingIntroCard/OnboardingIntroCard";
import { EditNameDialog } from "./EditNameDialog/EditNameDialog";

interface Props {
  name: string;
  isAwaitingGreeting: boolean;
  isGreetingFlow: boolean;
}

// The regular empty-session hero. It also stands in for the greeting page
// while the greeting is still generating, and in that mode the heading row
// is laid out exactly where OnboardingIntroCard is about to put it — same
// left edge, same type — so the swap leaves it where it already was
// instead of throwing it in from the centre of the page.
export function EmptyHero({ name, isAwaitingGreeting, isGreetingFlow }: Props) {
  return (
    <>
      <div
        className={cn(
          "mb-1 flex items-center gap-3",
          isGreetingFlow
            ? "w-full max-w-[48rem] justify-start"
            : "justify-center",
        )}
      >
        {isAwaitingGreeting && (
          <span className="relative size-8 shrink-0">
            <GlassOrb params={SMALL_ORB_PARAMS} />
          </span>
        )}
        <Text
          variant="h3"
          className={
            isGreetingFlow
              ? "!text-[1.25rem] text-zinc-800"
              : "!text-[1.375rem] text-zinc-700"
          }
        >
          Hey,{" "}
          {isGreetingFlow ? (
            <span style={{ color: ORB_PURPLE }}>{name}</span>
          ) : (
            <span className="text-violet-600">{name}</span>
          )}
          {/* The greeting page carries no name editing, so keeping the
              trigger here would pop it out mid-swap. */}
          {!isGreetingFlow && <EditNameDialog currentName={name} />}
        </Text>
      </div>
      {/* Held back with the composer: while the greeting decision is
          pending this line appearing then vanishing read as the page
          changing its mind on refresh. */}
      {!isAwaitingGreeting && (
        <TextGenerateEffect
          className="mb-8 !font-normal [&>div]:!mt-0 [&_div]:!text-[1.375rem] [&_div]:!leading-normal [&_div]:!tracking-normal"
          duration={0.6}
          words="Tell me about your work — I'll find what to automate."
        />
      )}
    </>
  );
}
