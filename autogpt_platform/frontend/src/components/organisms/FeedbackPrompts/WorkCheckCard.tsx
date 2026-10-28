"use client";

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
import { PromptCard } from "./components/PromptCard";
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
  task: string;
  eyebrow: string;
  initialRating?: Rating | null;
}

export function WorkCheckCard({
  name,
  avatarUrl,
  task,
  eyebrow,
  initialRating = null,
}: Props) {
  const [rating, setRating] = useState<Rating | null>(initialRating);
  const [isOpen, setIsOpen] = useState(true);

  if (!isOpen) return null;

  return (
    <PromptCard
      eyebrow={`${eyebrow} · ${task}`}
      title={
        rating && rating !== "great"
          ? `What should ${name} do better?`
          : `How did ${name} do?`
      }
      avatar={
        <ExpertAvatar
          name={name}
          avatarUrl={avatarUrl}
          size={40}
          className="ring-1 ring-border-base"
        />
      }
      onClose={() => setIsOpen(false)}
    >
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
          size="compact"
          thanks={`Thanks. ${name} learns from this.`}
        />
      ) : null}
    </PromptCard>
  );
}
