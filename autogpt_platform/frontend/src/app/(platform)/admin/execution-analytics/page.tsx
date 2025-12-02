import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { ExecutionAnalyticsForm } from "./components/ExecutionAnalyticsForm";

function ExecutionAnalyticsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Execution Analytics</h1>
            <p className="text-gray-500">
              Generate missing activity summaries and success scores for agent
              executions
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">Analytics Generation</h2>
          <p className="mb-6 text-gray-600">
            This tool will identify completed executions missing activity
            summaries or success scores and generate them using AI. Only
            executions that meet the criteria and are missing these fields will
            be processed.
          </p>

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
