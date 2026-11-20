import { useQuery } from "@tanstack/react-query";
import {
  CHANGELOG_TIMEOUT_MS,
  fetchLatestReleases,
  toRelease,
} from "./helpers";

export function useLoginWhatsNew() {
  const { data, isPending } = useQuery({
    queryKey: ["changelog", "latest"],
    queryFn: ({ signal }) => fetchWithTimeout(signal),
    select: (releases) => releases.map(toRelease),
    staleTime: 5 * 60 * 1000,
    retry: false,
    // Offline, the default "online" mode pauses the query before the timeout
    // starts, leaving the placeholder up. "always" fails fast to the fallback.
    networkMode: "always",
    refetchOnWindowFocus: false,
  });

  return {
    releases: data ?? [],
    isLoading: isPending,
  };
}

async function fetchWithTimeout(signal: AbortSignal) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CHANGELOG_TIMEOUT_MS);
  const cancel = () => controller.abort();
  signal.addEventListener("abort", cancel);
  try {
    return await fetchLatestReleases(controller.signal);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", cancel);
  }
}
