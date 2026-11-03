import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import Link from "next/link";
import { Text } from "@/components/atoms/Text/Text";
import {
  getGetV2GetLibraryAgentByGraphIdQueryKey,
  useGetV2GetLibraryAgentByGraphId,
} from "@/app/api/__generated__/endpoints/library/library";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useQueryStates, parseAsString } from "nuqs";
import { useBuilderTenantScope } from "@/app/(platform)/build/hooks/useBuilderTenantScope";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { getLibraryAgentHref } from "@/services/org-team/builder";

export const TriggerAgentBanner = () => {
  const tenantScope = useBuilderTenantScope();
  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });

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

  return (
    <Alert className="absolute bottom-4 left-1/2 z-20 w-auto -translate-x-1/2 rounded-xlarge select-none">
      <AlertTitle>You are building a Trigger Workflow</AlertTitle>
      <AlertDescription>
        Your workflow will listen for its trigger and will run when the time is
        right.
        <br />
        You can view its activity in your{" "}
        <Link
          href={
            libraryAgent
              ? getLibraryAgentHref(
                  libraryAgent.id,
                  libraryAgent.organization_id ?? tenantScope.organizationId,
                  libraryAgent.team_id ?? tenantScope.teamId,
                )
              : "/library"
          }
          className="underline"
        >
          <Text variant="body-tight" as="span" tone="inherit">
            Workflow Library
          </Text>
        </Link>
        .
      </AlertDescription>
    </Alert>
  );
};
