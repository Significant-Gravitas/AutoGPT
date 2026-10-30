"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import { BubbleChatIcon, Settings01Icon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { ExpertSidePanel } from "../../../components/ExpertSidePanel/ExpertSidePanel";
import { IdentityAvatar } from "../../../components/ExpertSidePanel/IdentityAvatar";
import { WelcomeQuestion } from "./components/WelcomeQuestion";
import { useExpertWelcome } from "./useExpertWelcome";

const ARRIVE_CLASS =
  "animate-in fade-in slide-in-from-bottom-1 duration-200 ease-out fill-mode-both motion-reduce:animate-none";

interface Props {
  expert: Expert;
  onClose: () => void;
  onStartChat: () => void;
  onOpenSettings: () => void;
}

/** The new expert's first words: who it thinks it is, read from its name and
 *  role, and the two or three things it needs to know to start. Opens with
 *  the page, so there is nothing to wait for and nothing to click first. */
export function ExpertWelcome({
  expert,
  onClose,
  onStartChat,
  onOpenSettings,
}: Props) {
  const welcome = useExpertWelcome(expert);
  const identity = {
    name: expert.name,
    avatarUrl: expert.avatar_url,
    color: expert.color,
  };
  const arrive = welcome.animateIntro ? ARRIVE_CLASS : undefined;

  return (
    <ExpertSidePanel
      identity={identity}
      title={expert.name}
      ariaLabel={`${expert.name} introduces themself`}
      panelId="welcome"
      closeLabel="Close"
      animateEntry={false}
      defaultWidth={460}
      onClose={onClose}
    >
      <div
        role="log"
        aria-live="polite"
        className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-5 pt-2 pb-6"
      >
        <ExpertLine avatar={identity}>
          Hi, I&apos;m {expert.name}.
          {welcome.intro ? (
            <span className={cn("block", arrive)}> {welcome.intro.intro}</span>
          ) : null}
        </ExpertLine>

        {welcome.intro ? (
          <div className={cn("flex flex-col gap-4 pl-10", arrive)}>
            {welcome.intro.questions.map((question, index) => (
              <WelcomeQuestion
                key={question.question}
                question={question.question}
                options={question.options}
                answer={welcome.answers[index]}
                disabled={welcome.phase !== "asking"}
                onAnswer={(value) => welcome.answer(index, value)}
              />
            ))}
            {welcome.phase === "asking" ? (
              <button
                type="button"
                onClick={welcome.skip}
                className="self-start rounded-md text-body text-foreground-tertiary underline-offset-2 outline-none hover:text-foreground-body hover:underline focus-visible:ring-2 focus-visible:ring-foreground-subtle"
              >
                Skip for now
              </button>
            ) : null}
          </div>
        ) : welcome.phase === "asking" ? (
          <TypingDots />
        ) : null}

        {welcome.phase !== "asking" ? (
          <div className={cn("flex flex-col gap-3", ARRIVE_CLASS)}>
            <ExpertLine avatar={identity}>
              {welcome.phase === "answered"
                ? "Got it. I'm writing that into my soul now."
                : "No problem. Tell me what you need whenever you're ready."}{" "}
              {budgetLine(expert)}
            </ExpertLine>
            <div className="flex flex-wrap gap-2 pl-10">
              <Button
                size="small"
                variant="primary"
                leadingIcon={BubbleChatIcon}
                onClick={onStartChat}
              >
                Give me a first task
              </Button>
              <Button
                size="small"
                variant="secondary"
                leadingIcon={Settings01Icon}
                onClick={onOpenSettings}
              >
                Open settings
              </Button>
            </div>
          </div>
        ) : null}
      </div>
    </ExpertSidePanel>
  );
}

function budgetLine(expert: Expert) {
  const budget = expert.weekly_budget;
  const opening =
    budget && budget > 0
      ? `I start on a ${creditsToUsdLabel(budget)} weekly budget.`
      : "I have no weekly budget cap.";
  return `${opening} Budget, voice, workflows and skills are all in Settings.`;
}

interface LineProps {
  avatar: { name: string; avatarUrl: string | null; color?: string | null };
  children: ReactNode;
}

function ExpertLine({ avatar, children }: LineProps) {
  return (
    <div className="flex items-start gap-3">
      <IdentityAvatar identity={avatar} className="size-7" imageSize={28} />
      <div className="min-w-0 flex-1 pt-0.5">
        <Text variant="body" tone="body">
          {children}
        </Text>
      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div
      role="status"
      aria-label="Typing"
      className="flex items-center gap-1 pl-10"
    >
      {[0, 1, 2].map((dot) => (
        <span
          key={dot}
          className="size-1.5 animate-pulse rounded-full bg-foreground-subtle motion-reduce:animate-none"
        />
      ))}
    </div>
  );
}
