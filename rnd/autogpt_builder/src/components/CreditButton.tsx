"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { IconRefresh } from "@/components/ui/icons";
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
        className="flex items-center space-x-2 text-muted-foreground"
      >
        <span>Credits: {credit}</span>
        <IconRefresh />
      </Button>
    )
  );
}
