import { withRoleAccess } from "@/lib/withRoleAccess";
import { AdminUsersView } from "./components/AdminUsersView/AdminUsersView";

export default async function AdminUsersPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminUsers = await withAdminAccess(AdminUsersView);
  return <ProtectedAdminUsers />;
}
