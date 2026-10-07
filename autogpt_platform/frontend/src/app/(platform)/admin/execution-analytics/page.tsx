import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { ExecutionAnalyticsForm } from "./components/ExecutionAnalyticsForm";
import { Text } from "@/components/atoms/Text/Text";

function ExecutionAnalyticsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <Text variant="h3" as="h1">
              Execution Analytics
            </Text>
            <Text variant="large" tone="muted">
              Generate missing activity summaries and success scores for agent
              executions
            </Text>
          </div>
        </div>

        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <Text variant="h4" as="h2" className="mb-4">
            Execution Analytics & Accuracy Monitoring
          </Text>
          <Text variant="large" tone="secondary" className="mb-6">
            Generate missing activity summaries and success scores for agent
            executions. After generation, accuracy trends and alerts will
            automatically be displayed to help monitor agent health over time.
          </Text>

          <Suspense
            fallback={<div className="py-10 text-center">Loading...</div>}
          >
            <ExecutionAnalyticsForm />
          </Suspense>
        </div>
      </div>
    </div>
  );
}

export default async function ExecutionAnalyticsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedExecutionAnalyticsDashboard = await withAdminAccess(
    ExecutionAnalyticsDashboard,
  );
  return <ProtectedExecutionAnalyticsDashboard />;
}
