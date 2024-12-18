"use client";

import { IconRefresh } from "@/components/ui/icons";
import { useState } from "react";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

interface CreditsCardProps {
  credits: number;
}

const CreditsCard = ({ credits }: CreditsCardProps) => {
  const [currentCredits, setCurrentCredits] = useState(credits);
  const api = useBackendAPI();

  const onRefresh = async () => {
    const { credits } = await api.getUserCredit("credits-card");
    setCurrentCredits(credits);
  };

  return (
    <div className="inline-flex h-[48px] items-center gap-2.5 rounded-2xl bg-neutral-200 p-4 dark:bg-neutral-800">
      <div className="flex items-center gap-0.5">
        <span className="p-ui-semibold text-base leading-7 text-neutral-900 dark:text-neutral-50">
          {currentCredits.toLocaleString()}
        </span>
        <span className="p-ui pl-1 text-base leading-7 text-neutral-900 dark:text-neutral-50">
          credits
        </span>
      </div>
      <Tooltip key="RefreshCredits" delayDuration={500}>
        <TooltipTrigger asChild>
          <button
            onClick={onRefresh}
            className="h-6 w-6 transition-colors hover:text-neutral-700 dark:hover:text-neutral-300"
            aria-label="Refresh credits"
          >
            <IconRefresh className="h-6 w-6" />
          </button>
        </TooltipTrigger>
        <TooltipContent>
          <p>Refresh credits</p>
        </TooltipContent>
      </Tooltip>
    </div>
  );
};

export default CreditsCard;
