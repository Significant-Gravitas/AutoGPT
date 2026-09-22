import { withRoleAccess } from "@/lib/withRoleAccess";
import { MemoryVisualizer } from "./components/MemoryVisualizer";

function MemoryDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">Memory Inspector</h1>
          <p className="text-gray-500">
            View entities, facts, and communities stored in your Graphiti memory
            graph. Trigger a community rebuild on demand.
          </p>
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
