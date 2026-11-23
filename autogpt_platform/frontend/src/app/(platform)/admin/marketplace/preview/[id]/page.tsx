import { withRoleAccess } from "@/lib/withRoleAccess";
import { AdminAgentPreview } from "./components/AdminAgentPreview/AdminAgentPreview";

export default async function AdminPreviewPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminPreview = await withAdminAccess(AdminAgentPreview);
  return <ProtectedAdminPreview />;
}
