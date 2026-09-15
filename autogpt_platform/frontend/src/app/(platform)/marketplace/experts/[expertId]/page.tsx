import { listExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { buildPageMetadata } from "@/lib/metadata";
import { Metadata } from "next";
import { ExpertPage } from "./components/ExpertPage";

export type MarketplaceExpertPageParams = { expertId: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<MarketplaceExpertPageParams>;
}): Promise<Metadata> {
  const params = await _params;
  const path = `/marketplace/experts/${params.expertId}`;
  const expert = await findExpertTemplate(params.expertId);

  if (!expert) {
    return buildPageMetadata({ title: "Expert - AutoGPT Marketplace", path });
  }

  // avatar_url points at an SVG, which no unfurler renders, so this stays a
  // text card until experts have a raster image.
  return buildPageMetadata({
    title: `${expert.name}, ${expert.role} - AutoGPT Marketplace`,
    description: expert.tagline || expert.bio,
    path,
    type: "profile",
  });
}

export default function MarketplaceExpertPage() {
  return <ExpertPage />;
}

async function findExpertTemplate(expertId: string): Promise<Expert | null> {
  try {
    const { data } = await listExpertTemplates();
    return (data as Expert[]).find((t) => t.id === expertId) ?? null;
  } catch {
    // Metadata must never break the page; the client fetch renders the error.
    return null;
  }
}
