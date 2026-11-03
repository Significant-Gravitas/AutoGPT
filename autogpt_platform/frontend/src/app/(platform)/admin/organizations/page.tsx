import { withRoleAccess } from "@/lib/withRoleAccess";
import { OrgAccessView } from "./components/OrgAccessView/OrgAccessView";

export default async function AdminOrganizationsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedOrgAccess = await withAdminAccess(OrgAccessView);
  return <ProtectedOrgAccess />;
}
