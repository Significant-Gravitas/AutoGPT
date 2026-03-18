import { withRoleAccess } from "@/lib/withRoleAccess";
import { AdminUsersPage } from "./components/AdminUsersPage/AdminUsersPage";

function AdminUsers() {
  return <AdminUsersPage />;
}

export default async function AdminUsersRoute() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminUsers = await withAdminAccess(AdminUsers);
  return <ProtectedAdminUsers />;
}
