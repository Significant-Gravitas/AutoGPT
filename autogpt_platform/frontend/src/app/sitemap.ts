import { listExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { getSiteUrl } from "@/lib/metadata";
import type { MetadataRoute } from "next";
import { PHASE_PRODUCTION_BUILD } from "next/constants";

// Static at build, re-rendered at most hourly, so experts published or
// archived later appear or disappear without a new deploy.
export const dynamic = "force-static";
export const revalidate = 3600;

const STATIC_PATHS = ["/marketplace"];

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = getSiteUrl();
  const paths = [...STATIC_PATHS, ...(await listPublicExpertPaths())];

  return paths.map((path) => ({ url: new URL(path, siteUrl).toString() }));
}

async function listPublicExpertPaths(): Promise<string[]> {
  try {
    const response = await listExpertTemplates();
    if (response.status !== 200) {
      throw new Error(
        `expert templates request failed with status ${response.status}`,
      );
    }
    return response.data
      .filter(isPublicTemplate)
      .map((template) => `/marketplace/experts/${template.id}`);
  } catch (error) {
    console.error("sitemap: could not list expert templates", error);
    // `next build` prerenders this route without a reachable backend, so the
    // build keeps the static pages. At runtime, throwing makes ISR keep
    // serving the last complete sitemap instead of caching a static-only one.
    if (isBuildPhase()) return [];
    throw error;
  }
}

function isBuildPhase(): boolean {
  return process.env.NEXT_PHASE === PHASE_PRODUCTION_BUILD;
}

function isPublicTemplate(template: ExpertTemplate): boolean {
  return template.is_template && !template.is_archived;
}
