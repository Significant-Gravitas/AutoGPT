import { okData } from "@/app/api/helpers";
import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";

export function useUserTimezone(): "not-set" | string | undefined {
  return useGetV1GetUserTimezone({
    query: { select: (res) => okData(res)?.timezone },
  }).data;
}
