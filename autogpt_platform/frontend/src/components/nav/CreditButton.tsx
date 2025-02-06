"use client";

import { Button } from "@/components/ui/button";
import { IconRefresh } from "@/components/ui/icons";
import useCredits from "@/hooks/useCredits";

export default function CreditButton() {
  const { credits, fetchCredits } = useCredits();

  return (
    credits !== null && (
      <Button
        onClick={fetchCredits}
        variant="outline"
        className="flex items-center space-x-2 rounded-xl bg-gray-200"
      >
        <span className="text-foreground mr-2 flex items-center">
          {credits} <span className="text-muted-foreground ml-2"> credits</span>
        </span>
        <IconRefresh />
      </Button>
    )
  );
}
