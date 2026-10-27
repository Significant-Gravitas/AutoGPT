"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";

const QUESTIONS = [
  {
    question: "What does chat usage cover?",
    answer:
      "Your chats with Otto and your experts, including the research and writing they do for you. It comes with your plan. A free trial has one allowance for the whole trial, and a paid plan resets daily and weekly.",
  },
  {
    question: "What are automation credits for?",
    answer:
      "Workflow runs and paid steps, like making images. You can buy credits, earn them by finishing setup steps or turn on auto-refill to top them up.",
  },
  {
    question: "When is my card charged?",
    answer:
      "When your plan starts and when you buy credits, including auto-refill. A chat or a step never charges your card directly.",
  },
  {
    question: "How do I get more?",
    answer:
      "For more chat usage, choose a bigger plan. For more automation credits, select Add credits in the wallet.",
  },
];

interface Props {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  /** Off on the billing page itself, where the link would go nowhere. */
  showBillingLink?: boolean;
}

export function UsageCreditsDialog({
  isOpen,
  onOpenChange,
  showBillingLink = true,
}: Props) {
  return (
    <Dialog
      title="How usage and credits work"
      variant="compact"
      controlled={{ isOpen, set: onOpenChange }}
      className="max-w-xl"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-5">
          <dl className="flex flex-col gap-4">
            {QUESTIONS.map((item) => (
              <div key={item.question} className="flex flex-col gap-1">
                <Text as="dt" variant="body-medium" tone="primary">
                  {item.question}
                </Text>
                <Text as="dd" variant="body" tone="body">
                  {item.answer}
                </Text>
              </div>
            ))}
          </dl>
          {showBillingLink && (
            <div className="flex justify-end">
              <Button
                as="NextLink"
                href="/settings/billing"
                variant="secondary"
                size="small"
                rightIcon={
                  <Icon icon={ArrowRight01Icon} size={16} aria-hidden />
                }
                onClick={() => onOpenChange(false)}
              >
                Go to billing settings
              </Button>
            </div>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
