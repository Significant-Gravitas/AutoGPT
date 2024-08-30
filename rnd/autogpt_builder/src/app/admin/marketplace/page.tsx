import { withRoleAccess } from "@/lib/withRoleAccess";

import React from "react";
import { getReviewableAgents } from "@/components/admin/marketplace/actions";
import AdminMarketplaceCard from "@/components/admin/marketplace/AdminMarketplaceCard";
import AdminMarketplaceAgentList from "@/components/admin/marketplace/AdminMarketplaceAgentList";
async function AdminMarketplace() {
  const agents = await getReviewableAgents();
  return (
    <div>
      <h3>Agents to review</h3>
      <AdminMarketplaceAgentList agents={agents.agents} />
    </div>
  );
}

export default async function AdminDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminMarketplace = await withAdminAccess(AdminMarketplace);
  return <ProtectedAdminMarketplace />;
}
