import { useTaskLiveUpdates } from "@/app/(platform)/tasks/data/useTaskLiveUpdates";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { WebSocketNotification } from "@/lib/autogpt-server-api/types";
import { useEffect } from "react";
import { getTaskToast } from "./helpers";

export function useTaskLiveUpdatesMount() {
  const api = useBackendAPI();
  useTaskLiveUpdates({ withToasts: false });

  useEffect(() => {
    function handle(n: WebSocketNotification) {
      const next = getTaskToast(n);
      if (next) toast(next);
    }
    return api.onWebSocketMessage("notification", handle);
  }, [api]);
}
