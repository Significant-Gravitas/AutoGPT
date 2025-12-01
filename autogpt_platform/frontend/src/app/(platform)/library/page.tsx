"use client";

import { ensureSupabaseClient } from "@/lib/supabase/hooks/helpers";
import { useEffect } from "react";
import FavoritesSection from "./components/FavoritesSection/FavoritesSection";
import LibraryActionHeader from "./components/LibraryActionHeader/LibraryActionHeader";
import LibraryAgentList from "./components/LibraryAgentList/LibraryAgentList";
import { LibraryPageStateProvider } from "./components/state-provider";

export default function LibraryPage() {
  useEffect(() => {
    document.title = "Library – AutoGPT Platform";
  }, []);

  // EXPERIMENTAL: Direct API call bypassing proxy
  useEffect(() => {
    async function makeDirectCall() {
      const supabaseClient = ensureSupabaseClient();
      if (!supabaseClient) {
        console.error("[EXPERIMENTAL] No Supabase client available");
        return;
      }

      const {
        data: { session },
      } = await supabaseClient.auth.getSession();

      if (!session?.access_token) {
        console.error("[EXPERIMENTAL] No session token available");
        return;
      }

      const baseUrl =
        process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8006/api";
      const url = `${baseUrl}/library/agents?page=1&page_size=8`;

      console.log("[EXPERIMENTAL] Making direct API call to:", url);
      const startTime = performance.now();

      try {
        const response = await fetch(url, {
          method: "GET",
          credentials: "include",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${session.access_token}`,
          },
        });

        const endTime = performance.now();
        const duration = endTime - startTime;
        console.log(
          `[EXPERIMENTAL] Direct API call completed in ${duration.toFixed(2)}ms`,
          {
            status: response.status,
            statusText: response.statusText,
            duration: `${duration.toFixed(2)}ms`,
          },
        );

        if (response.ok) {
          const data = await response.json();
          console.log("[EXPERIMENTAL] Direct API response:", data);
        } else {
          const errorText = await response.text();
          console.error("[EXPERIMENTAL] Direct API error:", errorText);
        }
      } catch (error) {
        const endTime = performance.now();
        const duration = endTime - startTime;
        console.error(
          `[EXPERIMENTAL] Direct API call failed after ${duration.toFixed(2)}ms:`,
          error,
        );
      }
    }

    void makeDirectCall();
  }, []);

  return (
    <main className="pt-160 container min-h-screen space-y-4 pb-20 pt-16 sm:px-8 md:px-12">
      <LibraryPageStateProvider>
        <LibraryActionHeader />
        <FavoritesSection />
        <LibraryAgentList />
      </LibraryPageStateProvider>
    </main>
  );
}
