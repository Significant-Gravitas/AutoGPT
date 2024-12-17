"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { IconRefresh } from "@/components/ui/icons";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

export default function CreditButton() {
  const [credit, setCredit] = useState<number | null>(null);
  const api = useBackendAPI();

  const fetchCredit = useCallback(async () => {
    try {
      const response = await api.getUserCredit();
      setCredit(response.credits);
    } catch (error) {
      console.error("Error fetching credit:", error);
      setCredit(null);
    }
  }, []);

  useEffect(() => {
    fetchCredit();
  }, [fetchCredit]);

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
