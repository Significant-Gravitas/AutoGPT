"use client";

import { useGetAdminGetUserIpReport } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { authClient } from "@/lib/auth/client";
import { useQuery } from "@tanstack/react-query";

export function useIPReport(userId: string) {
  const query = useGetAdminGetUserIpReport(userId, {
    query: { select: okData },
  });
  return {
    report: query.data,
    isLoading: query.isLoading,
    isForbidden: getStatus(query.error) === 403,
    refetch: query.refetch,
  };
}

// The address the signed-in admin's own session came from, so a dialog can
// warn before they block themselves.
export function useOwnSessionIP() {
  const query = useQuery({
    queryKey: ["admin", "own-session-ip"],
    queryFn: async () => {
      const { data, error } = await authClient.getSession();
      if (error) throw new Error(error.message || "Failed to read the session");
      return data?.session.ipAddress ?? null;
    },
    staleTime: Infinity,
  });
  return query.data ?? null;
}

function getStatus(error: unknown) {
  return typeof error === "object" && error !== null && "status" in error
    ? (error as { status: unknown }).status
    : undefined;
}
