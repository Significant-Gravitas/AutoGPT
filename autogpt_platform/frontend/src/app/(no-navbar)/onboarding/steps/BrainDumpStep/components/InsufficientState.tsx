"use client";

import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { cn } from "@/lib/utils";
import { offYourPlateExampleForRole } from "../prompts";
import { useIsNarrowScreen } from "../useIsNarrowScreen";
import { InsufficientActions } from "./InsufficientActions";
import {
  heardTextOf,
  ottoFirstLine,
  type InsufficientMode,
  type OttoReplyKind,
} from "./insufficient-copy";
import { OttoReply } from "./OttoReply";
import { YourMessage } from "./YourMessage";

interface Props {
  mode: InsufficientMode;
  // What Otto heard (voice) or read (typed); null when no words came through.
  heardText: string | null;
  // Which of Otto's fixed lines fits it; null for the default line.
  replyKind: OttoReplyKind | null;
  role: string;
  canRecord: boolean;
  onTalk: () => void;
  onWrite: () => void;
  onSkip: () => void;
}

// The take went through; it just didn't say enough to plan from. So the
// screen is the start of a conversation, not an error: what Otto heard, his
// answer, and one big, easy question. Otto stays still on purpose: the
// design system allows him a blink and no other motion.
export function InsufficientState({
  mode,
  heardText,
  replyKind,
  role,
  canRecord,
  onTalk,
  onWrite,
  onSkip,
}: Props) {
  const isNarrow = useIsNarrowScreen();
  const ottoSize = isNarrow ? 96 : 120;
  const heard = heardTextOf(heardText);

  return (
    <section className="relative isolate flex w-full max-w-xl flex-col items-center">
      <div
        aria-hidden="true"
        className="absolute inset-x-0 -top-10 -z-10 h-1/2 rounded-full bg-accent-surface blur-3xl"
      />
      <YourMessage text={heard} />
      <div
        className={cn("relative mt-2", isNarrow ? "size-24" : "size-[7.5rem]")}
      >
        <div
          aria-hidden="true"
          className="absolute -inset-[20%] rounded-full bg-brand/20 blur-2xl"
        />
        <AutopilotAvatar transparent size={ottoSize} className="relative" />
      </div>
      <div className="mt-4 w-full md:mt-5">
        <OttoReply firstLine={ottoFirstLine(mode, heard, replyKind)} />
      </div>
      <Text variant="body" tone="muted" className="mt-3 w-full text-left">
        Like:{" "}
        <Text variant="body" tone="secondary" italic as="span">
          “{offYourPlateExampleForRole(role)}”
        </Text>
      </Text>
      <div className="mt-5 w-full md:mt-6">
        <InsufficientActions
          mode={mode}
          canRecord={canRecord}
          onTalk={onTalk}
          onWrite={onWrite}
          onSkip={onSkip}
        />
      </div>
    </section>
  );
}
