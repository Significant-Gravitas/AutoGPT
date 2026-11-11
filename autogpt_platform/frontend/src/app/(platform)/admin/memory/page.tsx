import { withRoleAccess } from "@/lib/withRoleAccess";
import { MemoryInspector } from "./components/MemoryInspector";

export default async function MemoryDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(MemoryInspector);
  return <ProtectedDashboard />;
}
