"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { useState } from "react";
import { workflowNeedsSetup } from "../../../helpers";
import { CreateScheduleDialog } from "../CreateScheduleDialog";
import { ExpertAttentionCard } from "./ExpertAttentionCard";
import { useExpertNeedsYou } from "./useExpertNeedsYou";

interface Props {
  expert: Expert;
  enabled: boolean;
}

/** One card per item, styled like the stack sections in the chat sidebar. */
export function ExpertNeedsYouSection({ expert, enabled }: Props) {
  const { items, pendingIDs, decide } = useExpertNeedsYou({
    expertId: expert.id,
    enabled,
  });
  const [isSetupOpen, setIsSetupOpen] = useState(false);
  // "Needs setup" means a scheduled workflow with no schedule yet, so
  // finishing setup is creating that schedule.
  const workflowsNeedingSetup = expert.workflows.filter((workflow) =>
    workflowNeedsSetup(workflow),
  );

  if (items.length === 0) return null;

  return (
    <section aria-label="Needs you" className="flex min-w-0 flex-col">
      {/* No visible heading: the cards say what needs doing. The count is
          still announced for screen readers. */}
      <span role="status" className="sr-only">
        {`${items.length} ${items.length === 1 ? "item needs" : "items need"} your attention`}
      </span>
      <div className="flex flex-col gap-2">
        {items.map((item) => (
          <ExpertAttentionCard
            key={item.id}
            item={item}
            isProcessing={pendingIDs.has(item.id)}
            onDecision={decide}
            onFinishSetup={() => setIsSetupOpen(true)}
          />
        ))}
      </div>
      <CreateScheduleDialog
        expertId={expert.id}
        workflows={workflowsNeedingSetup}
        open={isSetupOpen}
        onClose={() => setIsSetupOpen(false)}
        title="Finish setup"
      />
    </section>
  );
}
