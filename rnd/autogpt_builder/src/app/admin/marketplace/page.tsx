import { withRoleAccess } from "@/lib/withRoleAccess";
import React from "react";

function AdminMarketplace() {
  return (
    <div>
      <h1>Admin Marketplace</h1>
      {/* Add your admin-only content here */}
    </div>
  );
}

export default async function AdminDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminMarketplace = await withAdminAccess(AdminMarketplace);
  return <ProtectedAdminMarketplace />;
}
