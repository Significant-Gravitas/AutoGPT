import { withRoleAccess } from "@/lib/withRoleAccess";

import React from "react";
// import { getReviewableAgents } from "@/components/admin/marketplace/actions";
// import AdminMarketplaceAgentList from "@/components/admin/marketplace/AdminMarketplaceAgentList";
// import AdminFeaturedAgentsControl from "@/components/admin/marketplace/AdminFeaturedAgentsControl";
import { Separator } from "@/components/ui/separator";
async function AdminMarketplace() {
  // const reviewableAgents = await getReviewableAgents();

  return (
    <>
      {/* <AdminMarketplaceAgentList agents={reviewableAgents.items} />
        <Separator className="my-4" />
        <AdminFeaturedAgentsControl className="mt-4" /> */}
    </>
  );
}

export default async function AdminDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminMarketplace = await withAdminAccess(AdminMarketplace);
  return <ProtectedAdminMarketplace />;
}
