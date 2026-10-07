"use client";

import { RunOutputs } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedRunView/components/RunOutputs";
import { okData } from "@/app/api/helpers";
import { useGetV1GetSharedExecution } from "@/app/api/__generated__/endpoints/default/default";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { useParams } from "next/navigation";
import { ShareActions } from "../components/ShareHeader/ShareActions";
import { ShareHeader } from "../components/ShareHeader/ShareHeader";

// Wraps the page in the shared header + a scrollable container.
// Header is the same component used by the chat share viewer so the
// two routes stay visually consistent.
function ExecutionShareChrome({
  title,
  children,
}: {
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen w-full flex-col bg-background">
      <ShareHeader title={title} actions={<ShareActions />} />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="container mx-auto px-4 py-8">{children}</div>
      </div>
    </div>
  );
}

export default function SharePage() {
  const params = useParams();
  const token = params.token as string;

  const {
    data: executionData,
    isLoading: loading,
    error,
  } = useGetV1GetSharedExecution(token, { query: { select: okData } });

  const is404 = !loading && !executionData;

  if (loading) {
    return (
      <ExecutionShareChrome>
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <LoadingSpinner size="large" className="mx-auto mb-4" />
            <Text variant="large" tone="muted">
              Loading shared execution...
            </Text>
          </div>
        </div>
      </ExecutionShareChrome>
    );
  }

  if (error || is404 || !executionData) {
    return (
      <ExecutionShareChrome>
        <div className="flex items-center justify-center py-16">
          <div className="mx-auto w-full max-w-md p-6">
            <Card className="border border-dashed border-zinc-300">
              <div className="space-y-4 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                  <Icon
                    icon={InformationCircleIcon}
                    className="h-6 w-6 text-muted-foreground"
                  />
                </div>
                <div className="space-y-2">
                  <Text variant="large-semibold" as="h3">
                    {is404 ? "Share Link Not Found" : "Unable to Load"}
                  </Text>
                  <Text variant="body" tone="muted">
                    {is404
                      ? "This shared link is invalid or has been disabled by the owner. Please check with the person who shared this link."
                      : "There was an error loading this shared execution. Please try refreshing the page."}
                  </Text>
                </div>
                <div className="pt-2">
                  <Button
                    variant="link"
                    onClick={() => window.location.reload()}
                  >
                    Try again
                  </Button>
                </div>
              </div>
            </Card>
            <Text variant="small" tone="muted" className="mt-8 text-center">
              Powered by AutoGPT Platform
            </Text>
          </div>
        </div>
      </ExecutionShareChrome>
    );
  }

  return (
    <ExecutionShareChrome title={executionData.graph_name}>
      <div className="mx-auto max-w-6xl">
        <div className="mb-6">
          <Alert>
            <AlertDescription>
              This is a publicly shared agent run result. The person who shared
              this link can disable access at any time.
            </AlertDescription>
          </Alert>
        </div>

        <Card className="mb-6">
          <div className="flex flex-col space-y-1.5">
            <Text variant="h4" as="h3" unmask={false}>
              {executionData.graph_name}
            </Text>
            {executionData.graph_description && (
              <Text
                variant="large"
                tone="muted"
                unmask={false}
                className="mt-2"
              >
                {executionData.graph_description}
              </Text>
            )}
          </div>
          <div className="mt-6">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Status:</span>
                <span className="ml-2 capitalize">
                  {executionData.status.toLowerCase()}
                </span>
              </div>
              <div>
                <span className="font-medium">Created:</span>
                <span className="ml-2">
                  {new Date(executionData.created_at).toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <Text variant="large-semibold" as="h3">
            Output
          </Text>
          <div className="mt-6">
            <RunOutputs outputs={executionData.outputs} shareToken={token} />
          </div>
        </Card>

        <Text variant="body" tone="muted" className="mt-8 text-center">
          Powered by AutoGPT Platform
        </Text>
      </div>
    </ExecutionShareChrome>
  );
}
