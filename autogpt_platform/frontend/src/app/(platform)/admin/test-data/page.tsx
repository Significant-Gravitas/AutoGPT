import { withRoleAccess } from "@/lib/withRoleAccess";
import { GenerateTestDataButton } from "./components/GenerateTestDataButton";

function TestDataDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Test Data Generation</h1>
            <p className="text-gray-500">
              Generate sample data for testing and development
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-xl font-semibold">Generate Test Data</h2>
          <p className="mb-6 text-gray-600">
            Use this tool to populate the database with sample test data. This
            is useful for development and testing purposes.
          </p>

          <div className="mb-6">
            <h3 className="mb-2 font-medium">Available Script Types:</h3>
            <ul className="list-inside list-disc space-y-2 text-gray-600">
              <li>
                <strong>E2E Test Data:</strong> Creates 15 test users with
                graphs, library agents, presets, store submissions, and API
                keys. Uses API functions for better compatibility.
              </li>
              <li>
                <strong>Full Test Data:</strong> Creates 100+ users with
                comprehensive test data including agent blocks, nodes,
                executions, analytics, and more. Takes longer to complete.
              </li>
            </ul>
          </div>

          <GenerateTestDataButton />
        </div>

        <div className="rounded-lg border bg-gray-50 p-6">
          <h3 className="mb-2 font-medium text-gray-700">
            What data is created?
          </h3>
          <div className="grid gap-4 text-sm text-gray-600 md:grid-cols-2">
            <div>
              <h4 className="font-medium">E2E Script:</h4>
              <ul className="mt-1 list-inside list-disc">
                <li>15 test users</li>
                <li>15 graphs per user</li>
                <li>Library agents</li>
                <li>Agent presets</li>
                <li>Store submissions</li>
                <li>API keys</li>
                <li>Creator profiles</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium">Full Script:</h4>
              <ul className="mt-1 list-inside list-disc">
                <li>100 users</li>
                <li>100 agent blocks</li>
                <li>Multiple graphs per user</li>
                <li>Agent nodes and links</li>
                <li>Graph executions</li>
                <li>Store listings and reviews</li>
                <li>Analytics data</li>
                <li>Credit transactions</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default async function TestDataDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedTestDataDashboard = await withAdminAccess(TestDataDashboard);
  return <ProtectedTestDataDashboard />;
}
