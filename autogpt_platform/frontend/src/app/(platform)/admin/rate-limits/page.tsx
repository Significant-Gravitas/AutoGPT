import { withRoleAccess } from "@/lib/withRoleAccess";
import { RateLimitManager } from "./components/RateLimitManager";

function RateLimitsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">User Rate Limits</h1>
          <p className="text-gray-500">
            Check and manage CoPilot rate limits per user
          </p>
        </div>
        <RateLimitManager />
      </div>
    </div>
  );
}

export default async function RateLimitsDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(RateLimitsDashboard);
  return <ProtectedDashboard />;
}
