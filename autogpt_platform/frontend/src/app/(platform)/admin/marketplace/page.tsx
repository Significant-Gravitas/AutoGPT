import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import type { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { AdminAgentsDataTable } from "./components/AdminAgentsDataTable";
import { AdminSkillSubmissions } from "./components/AdminSkillSubmissions";
import { Text } from "@/components/atoms/Text/Text";

type MarketplaceAdminPageSearchParams = {
  page?: string;
  status?: SubmissionStatus;
  search?: string;
};

async function AdminMarketplaceDashboard({
  searchParams,
}: {
  searchParams: MarketplaceAdminPageSearchParams;
}) {
  const page = searchParams.page ? Number.parseInt(searchParams.page) : 1;
  const status = searchParams.status;
  const search = searchParams.search;

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <Text variant="h3" as="h1">
              Marketplace Management
            </Text>
            <Text variant="large" tone="muted">
              Unified view for marketplace management and approval history
            </Text>
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

        <div className="mt-6 flex flex-col gap-3">
          <Text variant="h4" as="h2">
            Skill submissions
          </Text>
          <AdminSkillSubmissions />
        </div>
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
