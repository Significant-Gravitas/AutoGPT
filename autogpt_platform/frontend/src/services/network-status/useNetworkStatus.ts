"use client";

import { useEffect } from "react";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useNetworkStatus() {
  useEffect(function monitorNetworkStatus() {
    function handleOnline() {
      toast({
        title: "Connection restored",
        description: "You're back online",
        variant: "success",
      });
    }

    function handleOffline() {
      toast({
        title: "You're offline",
        description: "Check your internet connection",
        variant: "destructive",
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
