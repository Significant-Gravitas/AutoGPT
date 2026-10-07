import { withRoleAccess } from "@/lib/withRoleAccess";
import { RateLimitManager } from "./components/RateLimitManager";
import { Text } from "@/components/atoms/Text/Text";

function RateLimitsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <Text variant="h3" as="h1">
            User Rate Limits
          </Text>
          <Text variant="large" tone="muted">
            Check and manage CoPilot rate limits per user
          </Text>
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
