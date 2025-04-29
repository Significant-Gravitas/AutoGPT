import { AdminUserGrantHistory } from "@/components/admin/spending/admin-grant-history-data-table";
import type { CreditTransactionType } from "@/lib/autogpt-server-api";
import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";

function SpendingDashboard({
  searchParams,
}: {
  searchParams: {
    page?: string;
    status?: string;
    search?: string;
  };
}) {
  const page = searchParams.page ? Number.parseInt(searchParams.page) : 1;
  const search = searchParams.search;
  const status = searchParams.status as CreditTransactionType | undefined;

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
          <AdminUserGrantHistory
            initialPage={page}
            initialStatus={status}
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
    page?: string;
    status?: string;
    search?: string;
  };
}) {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedSpendingDashboard = await withAdminAccess(SpendingDashboard);
  return <ProtectedSpendingDashboard searchParams={searchParams} />;
}
