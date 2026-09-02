import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";

// Routes a selected global-search result to its destination. Chat sessions
// open on the copilot page via the ``sessionId`` query param so selection
// works from any page, not just copilot.
export function selectSearchResult(
  router: AppRouterInstance,
  item: SearchResultItem,
): void {
  switch (item.type) {
    case "chat_session":
      router.push(`/copilot?sessionId=${encodeURIComponent(item.id)}`);
      return;
    case "library_agent":
      router.push(`/library/agents/${item.id}`);
      return;
    case "store_agent": {
      // Store-agent rows carry creator + slug in ``metadata`` so we can build
      // the marketplace URL without an extra fetch.
      const metadata = (item.metadata ?? {}) as {
        creator?: string;
        slug?: string;
      };
      if (metadata.creator && metadata.slug) {
        router.push(
          `/marketplace/agent/${encodeURIComponent(metadata.creator)}/${encodeURIComponent(metadata.slug)}`,
        );
      }
      return;
    }
    case "workspace_file":
      // No dedicated viewer route — open the file's download URL in a new tab
      // so the user gets the content immediately.
      window.open(
        `/api/proxy/api/workspace/files/${item.id}/download`,
        "_blank",
        "noopener,noreferrer",
      );
      return;
  }
}
