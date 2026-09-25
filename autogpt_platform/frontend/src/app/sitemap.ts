import { listExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { getSiteUrl } from "@/lib/metadata";
import type { MetadataRoute } from "next";

// Static at build, re-rendered at most hourly, so experts published or
// archived after a deploy show up without one.
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
    // The crawler must still get the static pages when the API is down.
    console.error("sitemap: could not list expert templates", error);
    return [];
  }
}

function isPublicTemplate(template: ExpertTemplate): boolean {
  return template.is_template && !template.is_archived;
}
