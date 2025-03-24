import { AdminAgentsDataTable } from "@/components/admin/marketplace/admin-agents-data-table";
import { AdminUserGrantHistory } from "@/components/admin/spending/admin-grant-history-data-table";
import { AdminUserSpendingBalances } from "@/components/admin/spending/admin-spending-data-table";
import { SearchAndFilterAdminSpending } from "@/components/admin/spending/search-filter-form";
import type { SubmissionStatus } from "@/lib/autogpt-server-api";
import { withRoleAccess } from "@/lib/withRoleAccess";
import React, { Suspense } from "react";

function SpendingDashboard({
  searchParams,
}: {
  searchParams: {
    userPage?: string;
    grantPage?: string;
    search?: string;
  };
}) {
  const userPage = searchParams.userPage
    ? Number.parseInt(searchParams.userPage)
    : 1;
  const grantPage = searchParams.grantPage
    ? Number.parseInt(searchParams.grantPage)
    : 1;
  const search = searchParams.search;

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">User Spending</h1>
            <p className="text-gray-500">Manage user spending balances</p>
          </div>
        </div>

        <Suspense
          fallback={
            <div className="py-10 text-center">Loading submissions...</div>
          }
        >
          <SearchAndFilterAdminSpending
            initialUserPage={userPage}
            initialGrantPage={grantPage}
            initialSearch={search}
          />
          <AdminUserSpendingBalances
            initialPage={userPage}
            initialSearch={search}
          />
          <AdminUserGrantHistory
            initialPage={grantPage}
            initialSearch={search}
          />
        </Suspense>
      </div>
    </div>
  );
}

export default async function SpendingDashboardPage({
  searchParams,
}: {
  searchParams: {
    userPage?: string;
    grantPage?: string;
    search?: string;
  };
}) {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedSpendingDashboard = await withAdminAccess(SpendingDashboard);
  return <ProtectedSpendingDashboard searchParams={searchParams} />;
}
