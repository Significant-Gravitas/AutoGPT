import {
  getListExpertTemplatesQueryKey,
  listExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { buildPageMetadata } from "@/lib/metadata";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { notFound } from "next/navigation";
import { cache } from "react";
import { ExpertPage } from "./components/ExpertPage";

export const dynamic = "force-dynamic";

export type MarketplaceExpertPageParams = { expertId: string };

// Ads traffic lands here in bursts and the templates change rarely, so the
// server keeps one copy for a minute instead of asking the backend per view.
const TEMPLATES_REVALIDATE_SECONDS = 60;

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceExpertPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  const path = `/marketplace/experts/${params.expertId}`;
  const expert = findExpertTemplate(
    await fetchExpertTemplates(),
    params.expertId,
  );

  if (!expert) {
    return buildPageMetadata({ title: "Expert - AutoGPT Marketplace", path });
  }

  // avatar_url points at an SVG, which no unfurler renders, so this stays a
  // text card until experts have a raster image.
  return buildPageMetadata({
    title: `${expert.name}, ${expert.job_title || getExpertRoleLabel(expert.role)} · AI Expert - AutoGPT Marketplace`,
    description: expert.tagline || expert.bio,
    path,
    type: "profile",
  });
}

// Crawlers and the Ads landing-page check read the first response, so the
// expert's text has to be in the server HTML: the templates are fetched here
// and handed to the client tree, which hydrates useExpertPage with the expert
// on its first render instead of showing a skeleton. The route also sits
// outside the marketplace home's loading boundary (see ../(home)/loading.tsx):
// under one, the content would stream in a hidden chunk that only an inline
// script reveals, which is what a crawler without JavaScript never sees.
export default async function MarketplaceExpertPage({
  params: _params,
}: {
  params: Promise<MarketplaceExpertPageParams>;
}) {
  const { expertId } = await _params;
  const queryClient = getQueryClient();
  await queryClient.prefetchQuery({
    queryKey: getListExpertTemplatesQueryKey(),
    // Arg-less on purpose: React's cache keys on arguments, and react-query
    // would otherwise pass its context and miss generateMetadata's entry.
    queryFn: () => fetchExpertTemplatesOrThrow(),
  });
  const templates = queryClient.getQueryData<ExpertTemplatesResponse>(
    getListExpertTemplatesQueryKey(),
  );

  // Only a loaded list can say the id is unknown. When the backend was
  // unreachable nothing is dehydrated and the client fetch renders the error.
  if (templates && !findExpertTemplate(templates, expertId)) {
    notFound();
  }

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <ExpertPage />
    </HydrationBoundary>
  );
}

type ExpertTemplatesResponse = Awaited<ReturnType<typeof listExpertTemplates>>;

// One backend call per request, shared by generateMetadata and the page body.
const fetchExpertTemplatesOrThrow = cache(() =>
  listExpertTemplates(undefined, {
    next: { revalidate: TEMPLATES_REVALIDATE_SECONDS },
  }),
);

async function fetchExpertTemplates(): Promise<ExpertTemplatesResponse | null> {
  try {
    return await fetchExpertTemplatesOrThrow();
  } catch {
    // Metadata must never break the page; the client fetch renders the error.
    return null;
  }
}

function findExpertTemplate(
  templates: ExpertTemplatesResponse | null,
  expertId: string,
): Expert | null {
  if (!templates) return null;
  return (templates.data as Expert[]).find((t) => t.id === expertId) ?? null;
}
