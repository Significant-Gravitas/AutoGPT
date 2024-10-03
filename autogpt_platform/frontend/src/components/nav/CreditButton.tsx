"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { IconRefresh, IconCoin } from "@/components/ui/icons";
import AutoGPTServerAPI from "@/lib/autogpt-server-api";

export default function CreditButton() {
  const [credit, setCredit] = useState<number | null>(null);
  const api = new AutoGPTServerAPI();

  const fetchCredit = async () => {
    const response = await api.getUserCredit();
    setCredit(response.credits);
  };
  useEffect(() => {
    fetchCredit();
  }, [api]);

  return (
    credit !== null && (
      <Button
        onClick={fetchCredit}
        variant="outline"
        className="flex items-center space-x-2 rounded-xl bg-gray-200"
      >
        <span className="mr-2 flex items-center text-foreground">
          {credit} <span className="ml-2 text-muted-foreground"> credits</span>
        </span>
        <IconRefresh />
      </Button>
    )
  );
}
