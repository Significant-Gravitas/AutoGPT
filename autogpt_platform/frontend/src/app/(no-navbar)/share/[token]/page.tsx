"use client";

import React from "react";
import { useParams } from "next/navigation";
import { RunOutputs } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/SelectedRunView/components/RunOutputs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { InfoIcon } from "lucide-react";
import { useGetV1GetSharedExecution } from "@/app/api/__generated__/endpoints/default/default";

export default function SharePage() {
  const params = useParams();
  const token = params.token as string;

  const {
    data: response,
    isLoading: loading,
    error,
  } = useGetV1GetSharedExecution(token);

  const executionData = response?.status === 200 ? response.data : undefined;
  const is404 = !loading && !executionData;

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-primary"></div>
          <p className="text-muted-foreground">Loading shared execution...</p>
        </div>
      </div>
    );
  }

  if (error || is404 || !executionData) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="mx-auto w-full max-w-md p-6">
          <Card className="border-dashed">
            <CardContent className="pt-6">
              <div className="space-y-4 text-center">
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                  <InfoIcon className="h-6 w-6 text-muted-foreground" />
                </div>
                <div className="space-y-2">
                  <h3 className="text-lg font-semibold">
                    {is404 ? "Share Link Not Found" : "Unable to Load"}
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    {is404
                      ? "This shared link is invalid or has been disabled by the owner. Please check with the person who shared this link."
                      : "There was an error loading this shared execution. Please try refreshing the page."}
                  </p>
                </div>
                <div className="pt-2">
                  <button
                    onClick={() => window.location.reload()}
                    className="text-sm text-primary hover:underline"
                  >
                    Try again
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
          <div className="mt-8 text-center text-xs text-muted-foreground">
            <p>Powered by AutoGPT Platform</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl">
      <div className="mb-6">
        <Alert>
          <InfoIcon className="h-4 w-4" />
          <AlertDescription>
            This is a publicly shared agent run result. The person who shared
            this link can disable access at any time.
          </AlertDescription>
        </Alert>
      </div>

      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-2xl">{executionData.graph_name}</CardTitle>
          {executionData.graph_description && (
            <p className="mt-2 text-muted-foreground">
              {executionData.graph_description}
            </p>
          )}
        </CardHeader>
        <CardContent>
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
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Output</CardTitle>
        </CardHeader>
        <CardContent>
          <RunOutputs outputs={executionData.outputs} />
        </CardContent>
      </Card>

      <div className="mt-8 text-center text-sm text-muted-foreground">
        <p>Powered by AutoGPT Platform</p>
      </div>
    </div>
  );
}
