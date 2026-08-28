import {
  getGetV2GetLibraryAgentByGraphIdQueryKey,
  useGetV2GetLibraryAgentByGraphId,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { isValidUUID } from "@/lib/utils";
import Link from "next/link";
import { parseAsString, useQueryStates } from "nuqs";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { getLibraryAgentHref } from "@/services/org-team/builder";

export const WebhookDisclaimer = ({ nodeId }: { nodeId: string }) => {
  const tenantScope = useBuilderTenantScope();
  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

  // for a single agentId, we are fetching everything - need to make it better in the future
  const { data: libraryAgent } = useGetV2GetLibraryAgentByGraphId(
    flowID ?? "",
    {},
    {
      query: {
        select: (x) => {
          return x.data as LibraryAgent;
        },
        enabled: !!flowID && tenantScope.isReady,
        queryKey: getTeamScopedQueryKey(
          getGetV2GetLibraryAgentByGraphIdQueryKey(flowID ?? "", {}),
          tenantScope.organizationId,
          tenantScope.teamId,
        ),
      },
      request: getTenantRequestInit(
        tenantScope.organizationId,
        tenantScope.teamId,
        tenantScope.isReady,
      ),
    },
  );

  const isNodeSaved = isValidUUID(nodeId);

  return (
    <>
      <div className="px-4 pt-4">
        <Alert className="mb-3 rounded-xlarge">
          <AlertDescription>
            <Text variant="small-medium">
              You can set up and manage this trigger in your{" "}
              <Link
                href={
                  libraryAgent
                    ? getLibraryAgentHref(
                        libraryAgent.id,
                        libraryAgent.organization_id ??
                          tenantScope.organizationId,
                        libraryAgent.team_id ?? tenantScope.teamId,
                      )
                    : "/library"
                }
                className="underline"
              >
                Agent Library
              </Link>
              {!isNodeSaved && " (after saving the graph)"}.
            </Text>
          </AlertDescription>
        </Alert>
      </div>

      <Text variant="small" className="mb-4 ml-6 !text-purple-700">
        Below inputs are only for display purposes and cannot be edited.
      </Text>
    </>
  );
};
