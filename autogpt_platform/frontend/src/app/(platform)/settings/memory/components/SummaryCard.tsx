"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  scopeExpertID: string | null;
  scopeName: string;
  onViewSummary: () => void;
}

export function SummaryCard({
  scopeExpertID,
  scopeName,
  onViewSummary,
}: Props) {
  const isAutoPilot = scopeExpertID === null;
  return (
    <div className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)] sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Memory summary
        </Text>
        <Text variant="small" as="span" className="text-zinc-500">
          {isAutoPilot
            ? "Ask AutoPilot what it knows about you. Opens a chat — you can correct or forget anything from there."
            : `Ask ${scopeName} what they know about you and their work. Opens a chat.`}
        </Text>
      </div>
      <Button variant="primary" size="small" onClick={onViewSummary}>
        {isAutoPilot ? "View my summary" : `View ${scopeName}'s summary`}
      </Button>
    </div>
  );
}
