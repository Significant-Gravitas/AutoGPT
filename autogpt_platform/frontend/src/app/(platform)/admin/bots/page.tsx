import { withRoleAccess } from "@/lib/withRoleAccess";

import { BotsContent } from "./components/BotsContent";
import { Text } from "@/components/atoms/Text/Text";

function BotsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <Text variant="h3" as="h1">
            Bot Analytics
          </Text>
          <Text variant="large" tone="muted">
            Usage, reach and reliability across every live expert bot. No
            message content or user identity is collected — only aggregate
            counts and metrics.
          </Text>
        </div>

        <BotsContent />
      </div>
    </div>
  );
}

export default async function BotsPage() {
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(BotsDashboard);
  return <ProtectedDashboard />;
}
