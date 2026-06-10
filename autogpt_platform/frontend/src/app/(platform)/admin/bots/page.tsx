import { withRoleAccess } from "@/lib/withRoleAccess";

import { BotsContent } from "./components/BotsContent";

function BotsDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">Bot Analytics</h1>
          <p className="text-muted-foreground">
            Usage, reach and reliability across every live AutoPilot bot. No
            message content or user identity is collected — only aggregate
            counts and metrics.
          </p>
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
