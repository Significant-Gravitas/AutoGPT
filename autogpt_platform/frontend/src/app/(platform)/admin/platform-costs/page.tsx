import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { PlatformCostContent } from "./components/PlatformCostContent";

type SearchParams = {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
  model?: string;
  block_name?: string;
  tracking_type?: string;
  graph_exec_id?: string;
  page?: string;
  tab?: string;
};

function PlatformCostDashboard({
  searchParams,
}: {
  searchParams: SearchParams;
}) {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">Platform Costs</h1>
          <p className="text-muted-foreground">
            Track real API costs incurred by system credentials across providers
          </p>
        </div>

        <Suspense
          key={JSON.stringify(searchParams)}
          fallback={
            <div className="py-10 text-center">Loading cost data...</div>
          }
        >
          <PlatformCostContent searchParams={searchParams} />
        </Suspense>
      </div>
    </div>
  );
}

export default async function PlatformCostDashboardPage({
  searchParams,
}: {
  searchParams: Promise<SearchParams>;
}) {
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(PlatformCostDashboard);
  return <ProtectedDashboard searchParams={await searchParams} />;
}
