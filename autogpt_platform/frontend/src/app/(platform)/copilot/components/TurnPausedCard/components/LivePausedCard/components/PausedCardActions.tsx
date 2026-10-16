import { ArrowRight02Icon, Clock01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import { resetAt, resumeLabel } from "../../../helpers";
import type { useLivePausedCard } from "../useLivePausedCard";

interface Props {
  card: ReturnType<typeof useLivePausedCard>;
  onStop: () => void;
}

export function PausedCardActions({ card, onStop }: Props) {
  const { keep, resume, chatGpt } = card;
  // Continue on ChatGPT picks the task up too, once it has switched, so the
  // spinner follows the button that started it.
  const [pickUpVia, setPickUpVia] = useState<"keep_going" | "chatgpt">(
    "keep_going",
  );
  const isBusy =
    keep.isContinuing ||
    resume.isScheduling ||
    resume.isCancelling ||
    chatGpt.isSwitching;

  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-zinc-100 px-5 py-3">
      {resume.resumesAt && (
        <div className="flex items-center gap-1.5 pr-1">
          <Icon icon={Clock01Icon} size={14} className="text-purple-500" />
          <Text variant="small-medium" className="!text-zinc-700">
            Resumes {resetAt(resume.resumesAt, card.now)}
          </Text>
          <Button
            variant="link"
            onClick={() => void resume.cancelResume()}
            loading={resume.isCancelling}
            disabled={isBusy}
            className="ml-1 text-sm"
          >
            Cancel
          </Button>
        </div>
      )}
      {card.primary === "resume" && (
        <Button
          size="small"
          onClick={() => void resume.resumeAtReset()}
          loading={resume.isScheduling}
          disabled={isBusy}
        >
          {resumeLabel(card.shown.resets_at, card.now)}
        </Button>
      )}
      {card.primary === "keep_going" && (
        <Button
          size="small"
          onClick={() => {
            setPickUpVia("keep_going");
            void keep.keepGoing();
          }}
          loading={keep.isContinuing && pickUpVia === "keep_going"}
          disabled={isBusy}
        >
          Keep going
        </Button>
      )}
      {card.chatGptName && (
        <Button
          size="small"
          variant="secondary"
          onClick={() => {
            setPickUpVia("chatgpt");
            void chatGpt.continueHere();
          }}
          loading={
            chatGpt.isSwitching ||
            (keep.isContinuing && pickUpVia === "chatgpt")
          }
          disabled={isBusy}
        >
          Continue on {card.chatGptName}
        </Button>
      )}
      {card.canStop && (
        <Button size="small" variant="ghost" onClick={onStop} disabled={isBusy}>
          Stop here
        </Button>
      )}
      {card.canUpgrade && (
        <Button
          size="small"
          // The only way on for a trial or a budget that never refills.
          variant={
            card.primary === null && !resume.resumesAt ? "primary" : "ghost"
          }
          onClick={card.upgrade}
          disabled={isBusy || card.isUpgradeDisabled}
          rightIcon={<Icon icon={ArrowRight02Icon} size={14} />}
        >
          Upgrade
        </Button>
      )}
    </div>
  );
}
