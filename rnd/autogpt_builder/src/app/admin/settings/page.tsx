import { withRoleAccess } from "@/lib/withRoleAccess";
import React from "react";

function AdminSettings() {
  return (
    <div>
      <h1>Admin Settings</h1>
      {/* Add your admin-only settings content here */}
    </div>
  );
}

export default async function AdminSettingsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminSettings = await withAdminAccess(AdminSettings);
  return <ProtectedAdminSettings />;
}
