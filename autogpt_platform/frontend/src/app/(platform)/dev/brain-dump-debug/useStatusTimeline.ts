import { useGetBrainDumpStatus } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useEffect, useState } from "react";
import { STATUS_POLL_MS } from "./helpers";
import type { StatusSeenAt } from "./waterfall";

interface Args {
  enabled: boolean;
}

export function useStatusTimeline({ enabled }: Args) {
  const { data, isLoading, isError, error, refetch } = useGetBrainDumpStatus({
    query: { enabled, refetchInterval: STATUS_POLL_MS },
  });

  const [seenAt, setSeenAt] = useState<StatusSeenAt>({});

  const dump = data && data.status === 200 ? data.data : null;
  const status = dump?.status ?? null;

  useEffect(() => {
    if (!status) return;
    setSeenAt((current) =>
      current[status] === undefined
        ? { ...current, [status]: Date.now() }
        : current,
    );
  }, [status]);

  return { dump, seenAt, isLoading, isError, error, refetch };
}
