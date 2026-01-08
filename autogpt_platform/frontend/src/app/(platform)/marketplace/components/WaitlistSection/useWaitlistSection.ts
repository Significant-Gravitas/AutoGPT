"use client";

import { useEffect, useState } from "react";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { StoreWaitlistEntry } from "@/lib/autogpt-server-api/types";

export function useWaitlistSection() {
  const [waitlists, setWaitlists] = useState<StoreWaitlistEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    async function fetchWaitlists() {
      try {
        const api = new BackendAPI();
        const response = await api.getWaitlists();
        setWaitlists(response.listings);
      } catch (error) {
        console.error("Error fetching waitlists:", error);
        setHasError(true);
      } finally {
        setIsLoading(false);
      }
    }

    fetchWaitlists();
  }, []);

  return { waitlists, isLoading, hasError };
}
