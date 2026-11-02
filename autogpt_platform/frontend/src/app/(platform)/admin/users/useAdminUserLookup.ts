"use client";

import { authClient } from "@/lib/auth/client";
import { useQuery } from "@tanstack/react-query";
import { useQueryState } from "nuqs";
import {
  type AdminUser,
  type LookupQuery,
  parseLookupQuery,
  toAdminUser,
  USER_QUERY_PARAM,
} from "./helpers";

export function useAdminUserLookup() {
  const [userParam, setUserParam] = useQueryState(USER_QUERY_PARAM);
  const query = parseLookupQuery(userParam);

  const lookup = useQuery({
    queryKey: getAdminUserLookupQueryKey(query),
    queryFn: () => lookUpUser(query!),
    enabled: query !== null,
  });

  function search(raw: string) {
    void setUserParam(raw.trim() || null);
  }

  return {
    rawQuery: userParam ?? "",
    query,
    user: lookup.data ?? null,
    isLoading: lookup.isLoading,
    isNotFound: lookup.isSuccess && lookup.data === null,
    error: lookup.error,
    search,
    refetch: lookup.refetch,
  };
}

export function getAdminUserLookupQueryKey(query: LookupQuery | null) {
  return ["admin", "user-lookup", query?.kind, query?.value] as const;
}

// Both paths are exact: get-user matches the primary key, and list-users with
// filterOperator "eq" compiles to `=`, so `_` and `%` match only themselves.
async function lookUpUser(query: LookupQuery): Promise<AdminUser | null> {
  if (query.kind === "id") {
    const { data, error } = await authClient.admin.getUser({
      query: { id: query.value },
    });
    if (error?.status === 404) return null;
    if (error) throw new Error(error.message || "Failed to look up the user");
    return data ? toAdminUser(data) : null;
  }

  const { data, error } = await authClient.admin.listUsers({
    query: {
      filterField: "email",
      filterOperator: "eq",
      filterValue: query.value,
      limit: 1,
    },
  });
  if (error) throw new Error(error.message || "Failed to look up the user");
  const match = data?.users[0];
  return match ? toAdminUser(match) : null;
}
