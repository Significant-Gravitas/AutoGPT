import { withRoleAccess } from "@/lib/withRoleAccess";
import React from "react";

function AdminDashboard() {
  return (
    <div>
      <h1>Admin Dashboard</h1>
      {/* Add your admin-only content here */}
    </div>
  );
}

export default async function AdminDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminDashboard = await withAdminAccess(AdminDashboard);
  return <ProtectedAdminDashboard />;
}
