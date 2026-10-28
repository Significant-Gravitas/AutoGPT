import { fetchChangelogEntries } from "@/components/molecules/ChangelogPopup/helpers";
import { useQuery } from "@tanstack/react-query";
import { CHANGELOG_TIMEOUT_MS, RELEASE_COUNT, toRelease } from "./helpers";

export function useLoginWhatsNew() {
  const { data, isPending } = useQuery({
    queryKey: ["changelog", "index"],
    queryFn: ({ signal }) => fetchWithTimeout(signal),
    select: (entries) => entries.slice(0, RELEASE_COUNT).map(toRelease),
    staleTime: 5 * 60 * 1000,
    retry: false,
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
    return await fetchChangelogEntries(controller.signal);
  } finally {
    clearTimeout(timer);
    signal.removeEventListener("abort", cancel);
  }
}
