"use client";

import { useEffect } from "react";
import { toast } from "sonner";

export function useNetworkStatus() {
  useEffect(function monitorNetworkStatus() {
    function handleOnline() {
      toast.success("Connection restored", {
        description: "You're back online",
      });
    }

    function handleOffline() {
      toast.error("You're offline", {
        description: "Check your internet connection",
      });
    }

    window.addEventListener("online", handleOnline);
    window.addEventListener("offline", handleOffline);

    return function cleanup() {
      window.removeEventListener("online", handleOnline);
      window.removeEventListener("offline", handleOffline);
    };
  }, []);
}
