import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import type { SubmissionStatus } from "@/lib/autogpt-server-api/types";
import { AdminAgentsDataTable } from "@/components/admin/marketplace/admin-agents-data-table";

type MarketplaceAdminPageSearchParams = {
  page?: string;
  status?: string;
  search?: string;
};

async function AdminMarketplaceDashboard({
  searchParams,
}: {
  searchParams: MarketplaceAdminPageSearchParams;
}) {
  const page = searchParams.page ? Number.parseInt(searchParams.page) : 1;
  const status = searchParams.status as SubmissionStatus | undefined;
  const search = searchParams.search;

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Marketplace Management</h1>
            <p className="text-gray-500">
              Unified view for marketplace management and approval history
            </p>
          </div>
        </div>

        <Suspense
          fallback={
            <div className="py-10 text-center">Loading submissions...</div>
          }
        >
          <AdminAgentsDataTable
            initialPage={page}
            initialStatus={status}
            initialSearch={search}
          />
        </Suspense>
      </div>
    </div>
  );
}

export default async function AdminMarketplacePage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceAdminPageSearchParams>;
}) {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminMarketplace = await withAdminAccess(
    AdminMarketplaceDashboard,
  );
  return <ProtectedAdminMarketplace searchParams={await searchParams} />;
}
