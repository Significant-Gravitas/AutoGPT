import { listExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { getSiteUrl } from "@/lib/metadata";
import type { MetadataRoute } from "next";

// Rendered on request: the frontend image is built without a reachable
// backend, so a prerendered sitemap would ship without expert pages and
// serve that for its whole first revalidation window. The expert list is
// instead cached by fetch for an hour once it has loaded successfully, so
// experts published or archived later appear or disappear without a deploy.
export const dynamic = "force-dynamic";

const EXPERT_LIST_REVALIDATE_SECONDS = 3600;
const STATIC_PATHS = ["/marketplace"];

export default async function sitemap(): Promise<MetadataRoute.Sitemap> {
  const siteUrl = getSiteUrl();
  const paths = [...STATIC_PATHS, ...(await listPublicExpertPaths())];

  return paths.map((path) => ({ url: new URL(path, siteUrl).toString() }));
}

async function listPublicExpertPaths(): Promise<string[]> {
  try {
    const response = await listExpertTemplates(undefined, {
      next: { revalidate: EXPERT_LIST_REVALIDATE_SECONDS },
    });
    if (response.status !== 200) {
      console.error(
        "sitemap: expert templates request failed",
        response.status,
      );
      return [];
    }
    return response.data
      .filter(isPublicTemplate)
      .map((template) => `/marketplace/experts/${template.id}`);
  } catch (error) {
    // Only 200 responses enter the fetch cache, so an outage is never kept:
    // the crawler still gets the static pages and the next request retries.
    console.error("sitemap: could not list expert templates", error);
    return [];
  }
}

function isPublicTemplate(template: ExpertTemplate): boolean {
  return template.is_template && !template.is_archived;
}
