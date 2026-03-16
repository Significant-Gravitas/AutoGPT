import { withRoleAccess } from "@/lib/withRoleAccess";
import { AdminCopilotPage } from "./components/AdminCopilotPage/AdminCopilotPage";

function AdminCopilot() {
  return <AdminCopilotPage />;
}

export default async function AdminCopilotRoute() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminCopilot = await withAdminAccess(AdminCopilot);
  return <ProtectedAdminCopilot />;
}
