"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import {
  Happy01Icon,
  NeutralIcon,
  Sad01Icon,
} from "@hugeicons/core-free-icons";
import { motion } from "framer-motion";
import { useState } from "react";
import { ReasonFlow } from "./components/ReasonFlow";
import { ThanksLine } from "./components/ThanksLine";
import { WORK_REASONS } from "./helpers";

type Rating = "bad" | "okay" | "great";

const RATINGS = [
  {
    value: "bad" as const,
    label: "Not good",
    icon: Sad01Icon,
    tone: "hover:bg-error-surface hover:text-error-foreground",
  },
  {
    value: "okay" as const,
    label: "Okay",
    icon: NeutralIcon,
    tone: "hover:bg-warning-surface hover:text-warning-foreground",
  },
  {
    value: "great" as const,
    label: "Great",
    icon: Happy01Icon,
    tone: "hover:bg-success-surface hover:text-success-foreground",
  },
];

interface Props {
  name: string;
  avatarUrl: string | null;
  note: string;
  initialRating?: Rating | null;
}

// Sits at the end of a chat, under the reply that finished the work. Otto
// asks for it when a turn delivers what was asked; the backend decides
// whether it shows (the first time, then at most once a week).
export function WorkCheckCard({
  name,
  avatarUrl,
  note,
  initialRating = null,
}: Props) {
  const [rating, setRating] = useState<Rating | null>(initialRating);
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <div className="flex max-w-xl flex-col gap-4 rounded-2xl border border-border-base bg-surface p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <ExpertAvatar
          name={name}
          avatarUrl={avatarUrl}
          size={32}
          className="ring-1 ring-border-base"
        />
        <div className="flex flex-1 flex-col gap-0.5">
          <Text variant="body-medium" className="!text-foreground-strong">
            {rating && rating !== "great"
              ? `What should ${name} do better?`
              : `How did ${name} do?`}
          </Text>
          <Text variant="small" className="!text-foreground-tertiary">
            {note}
          </Text>
        </div>
        <Button
          variant="ghost"
          size="xs"
          className="-mr-1 -mt-1"
          onClick={() => setIsOpen(false)}
        >
          Not now
        </Button>
      </div>
      {rating === null ? (
        <div className="grid grid-cols-3 gap-2">
          {RATINGS.map((option) => (
            <motion.button
              key={option.value}
              type="button"
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setRating(option.value)}
              className={cn(
                "flex flex-col items-center gap-1.5 rounded-xl bg-surface-muted py-3 text-foreground-secondary transition-colors",
                option.tone,
              )}
            >
              <Icon icon={option.icon} size={28} strokeWidth={1.75} />
              <Text variant="small-medium" className="!text-inherit">
                {option.label}
              </Text>
            </motion.button>
          ))}
        </div>
      ) : null}
      {rating === "great" ? (
        <ThanksLine withChangelog={false}>
          {`Glad to hear it. We'll pass that on to ${name}.`}
        </ThanksLine>
      ) : null}
      {rating === "bad" || rating === "okay" ? (
        <ReasonFlow
          reasons={WORK_REASONS}
          label={`What should ${name} do better?`}
          size="compact"
          thanks={`Thanks. ${name} learns from this.`}
        />
      ) : null}
    </div>
  );
}
