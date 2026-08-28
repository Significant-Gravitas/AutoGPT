import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { getTenantRequestInit } from "@/components/contextual/TeamPicker/helpers";
import {
  getCopilotHref,
  getLibraryAgentHref,
} from "@/services/org-team/builder";
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
      router.push(
        getCopilotHref(
          item.id,
          item.organization_id ?? null,
          item.team_id ?? null,
        ),
      );
      return;
    case "library_agent":
      router.push(
        getLibraryAgentHref(
          item.id,
          item.organization_id ?? null,
          item.team_id ?? null,
        ),
      );
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
      void openWorkspaceFile(
        item.id,
        item.organization_id ?? null,
        item.team_id ?? null,
      );
      return;
  }
}

async function openWorkspaceFile(
  fileId: string,
  organizationId: string | null,
  teamId: string | null,
) {
  const target = window.open("about:blank", "_blank", "noopener,noreferrer");
  const response = await fetch(
    `/api/proxy/api/workspace/files/${encodeURIComponent(fileId)}/download`,
    getTenantRequestInit(organizationId, teamId),
  );
  if (!response.ok) {
    target?.close();
    return;
  }
  const objectUrl = URL.createObjectURL(await response.blob());
  target?.location.replace(objectUrl);
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
}
