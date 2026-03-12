import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { useQuery } from "@tanstack/react-query";

interface UsageWindow {
  used: number;
  limit: number;
  resets_at: string;
}

export interface CoPilotUsageStatus {
  session: UsageWindow;
  weekly: UsageWindow;
}

export function useUsageLimits(sessionID: string | null) {
  return useQuery({
    queryKey: ["copilot-usage", sessionID],
    queryFn: async (): Promise<CoPilotUsageStatus> => {
      const { token } = await getWebSocketToken();

      const params = new URLSearchParams();
      if (sessionID) {
        params.set("session_id", sessionID);
      }

      const res = await fetch(
        `${environment.getAGPTServerBaseUrl()}/api/chat/usage?${params}`,
        {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        },
      );

      if (!res.ok) {
        throw new Error(`Failed to fetch usage: ${res.status}`);
      }

      return res.json();
    },
    enabled: !!sessionID,
    refetchInterval: 30000,
    staleTime: 10000,
  });
}
