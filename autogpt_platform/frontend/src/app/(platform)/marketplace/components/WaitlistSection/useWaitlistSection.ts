"use client";

import { useEffect, useState } from "react";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { StoreWaitlistEntry } from "@/lib/autogpt-server-api/types";
import { useSupabaseStore } from "@/lib/supabase/hooks/useSupabaseStore";

export function useWaitlistSection() {
  const { user } = useSupabaseStore();
  const [waitlists, setWaitlists] = useState<StoreWaitlistEntry[]>([]);
  const [joinedWaitlistIds, setJoinedWaitlistIds] = useState<Set<string>>(
    new Set(),
  );
  const [isLoading, setIsLoading] = useState(true);
  const [hasError, setHasError] = useState(false);

  useEffect(() => {
    async function fetchData() {
      try {
        const api = new BackendAPI();

        // Fetch waitlists
        const response = await api.getWaitlists();
        setWaitlists(response.listings);

        // Fetch memberships if logged in
        if (user) {
          try {
            const memberships = await api.getMyWaitlistMemberships();
            setJoinedWaitlistIds(new Set(memberships));
          } catch (error) {
            // Don't fail the whole component if membership fetch fails
            console.error("Error fetching waitlist memberships:", error);
          }
        }
      } catch (error) {
        console.error("Error fetching waitlists:", error);
        setHasError(true);
      } finally {
        setIsLoading(false);
      }
    }

    fetchData();
  }, [user]);

  // Function to add a waitlist ID to joined set (called after successful join)
  function markAsJoined(waitlistId: string) {
    setJoinedWaitlistIds((prev) => new Set([...prev, waitlistId]));
  }

  return { waitlists, joinedWaitlistIds, isLoading, hasError, markAsJoined };
}
