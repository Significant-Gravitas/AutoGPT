import { withRoleAccess } from "@/lib/withRoleAccess";
import { MemoryVisualizer } from "./components/MemoryVisualizer";
import { Text } from "@/components/atoms/Text/Text";

function MemoryDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <Text variant="h3" as="h1">
            Memory Inspector
          </Text>
          <Text variant="large" tone="muted">
            View entities, facts, and communities stored in your Graphiti memory
            graph. Trigger a community rebuild on demand.
          </Text>
        </div>
        <MemoryVisualizer />
      </div>
    </div>
  );
}

export default async function MemoryDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(MemoryDashboard);
  return <ProtectedDashboard />;
}
