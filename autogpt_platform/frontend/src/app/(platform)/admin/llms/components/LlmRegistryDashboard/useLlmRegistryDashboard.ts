"use client";

import {
  useGetV2AdminListModels,
  useGetV2AdminListProviders,
  useGetV2ListCreators,
  useGetV2ListMigrations,
  useGetV2ListRoutes,
  useGetV2ListRouteWarnings,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";

export function useLlmRegistryDashboard() {
  // NOTE: the `select: okData` option must be inlined per call — hoisting it
  // to a shared const collapses okData's generic to `any` and loses types.
  const models = useGetV2AdminListModels(
    { page: 1, page_size: 200 },
    { query: { select: okData } },
  );
  const providers = useGetV2AdminListProviders({ query: { select: okData } });
  const creators = useGetV2ListCreators({ query: { select: okData } });
  const migrations = useGetV2ListMigrations(
    { include_reverted: true },
    { query: { select: okData } },
  );
  const routes = useGetV2ListRoutes({ query: { select: okData } });
  const routeWarnings = useGetV2ListRouteWarnings({
    query: { select: okData },
  });

  return {
    models,
    providers,
    creators,
    migrations,
    routes,
    routeWarnings,
  };
}
