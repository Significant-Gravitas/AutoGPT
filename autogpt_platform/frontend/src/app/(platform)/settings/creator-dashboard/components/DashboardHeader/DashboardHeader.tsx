"use client";

import { PlusIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";

import { EASE_OUT } from "../../helpers";

interface PublishState {
  isOpen: boolean;
  step: "select" | "info" | "review";
  submissionData:
    | import("@/app/api/__generated__/models/storeSubmission").StoreSubmission
    | null;
}

interface Props {
  publishState: PublishState;
  onPublishStateChange: (state: PublishState) => void;
  onOpenSubmit: () => void;
}

export function DashboardHeader({
  publishState,
  onPublishStateChange,
  onOpenSubmit,
}: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <motion.header
      initial={reduceMotion ? false : { opacity: 0, y: 8 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={reduceMotion ? undefined : { duration: 0.32, ease: EASE_OUT }}
      className="flex flex-col gap-5 pb-2 pl-4 pr-1 md:flex-row md:items-end md:justify-between"
    >
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          Creator dashboard
        </Text>
        <Text variant="body" className="mt-3 max-w-[640px] text-zinc-700">
          Track your store submissions, see how they perform, and ship updates
          for the agents you publish.
        </Text>
      </div>

      <PublishAgentModal
        targetState={publishState}
        onStateChange={onPublishStateChange}
        trigger={
          <Button
            data-testid="submit-agent-button"
            size="large"
            onClick={onOpenSubmit}
            leftIcon={<PlusIcon size={18} weight="bold" />}
          >
            Submit agent
          </Button>
        }
      />
    </motion.header>
  );
}
