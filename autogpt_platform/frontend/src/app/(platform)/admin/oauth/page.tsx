import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { OAuthAppList } from "./components/OAuthAppList";
import { getOAuthApps } from "./actions";

type OAuthPageSearchParams = {
  page?: string;
  search?: string;
};

async function OAuthDashboard({
  searchParams,
}: {
  searchParams: OAuthPageSearchParams;
}) {
  const page = searchParams.page ? Number.parseInt(searchParams.page) : 1;
  const search = searchParams.search;

  const data = await getOAuthApps(page, 20, search);

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">OAuth Applications</h1>
            <p className="text-gray-500">
              Manage OAuth applications for third-party integrations
            </p>
          </div>
        </div>

        <Suspense
          key={`${page}-${search}`}
          fallback={
            <div className="py-10 text-center">Loading OAuth applications...</div>
          }
        >
          <OAuthAppList initialData={data} initialSearch={search} />
        </Suspense>
      </div>
    </div>
  );
}

export default async function OAuthDashboardPage({
  searchParams,
}: {
  searchParams: Promise<OAuthPageSearchParams>;
}) {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedOAuthDashboard = await withAdminAccess(OAuthDashboard);
  return <ProtectedOAuthDashboard searchParams={await searchParams} />;
}
