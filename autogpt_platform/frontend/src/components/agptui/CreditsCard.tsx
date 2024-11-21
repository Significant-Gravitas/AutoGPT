"use client";

import { IconRefresh } from "@/components/ui/icons";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useState } from "react";

interface CreditsCardProps {
  credits: number;
}

const CreditsCard = ({ credits }: CreditsCardProps) => {
  const [currentCredits, setCurrentCredits] = useState(credits);
  const api = new AutoGPTServerAPI();

  const onRefresh = async () => {
    const { credits } = await api.getUserCredit();
    setCurrentCredits(credits);
  };

  return (
    <div className="inline-flex h-[60px] items-center gap-2.5 rounded-2xl bg-neutral-200 dark:bg-neutral-800 p-4">
      <div className="flex items-center gap-0.5">
        <span className="font-['Poppins'] text-base font-semibold leading-7 text-neutral-900 dark:text-neutral-50">
          {currentCredits.toLocaleString()}
        </span>
        <span className="font-['Poppins'] pl-1 text-base font-semibold leading-7 text-neutral-900 dark:text-neutral-50">
          credits
        </span>
      </div>
      <button
        onClick={onRefresh}
        className="h-6 w-6 transition-colors hover:text-neutral-700"
        aria-label="Refresh credits"
      >
        <IconRefresh className="h-6 w-6" />
      </button>
    </div>
  );
};

export default CreditsCard;
