import { ADMIN_PAGES, PROTECTED_PAGES } from "@/lib/auth/helpers";
import { getSiteUrl } from "@/lib/metadata";
import type { MetadataRoute } from "next";

const PRIVATE_PREFIXES = ["/api", "/auth"];

// Marketplace pages render their bodies from browser fetches through
// /api/proxy (expert profiles, search, skills) and load store media through
// the same proxy or its /api/store/media rewrite. Crawlers apply robots.txt
// to those rendering fetches too, so a blanket /api disallow would leave them
// indexing a loading shell. Only the anonymous, read-only marketplace routes
// are opened; longest-match wins, so these beat the /api disallow.
const PUBLIC_API_PREFIXES = [
  "/api/store/media/",
  "/api/proxy/api/store/media/",
  "/api/proxy/api/store/agents",
  "/api/proxy/api/store/creators",
  "/api/proxy/api/store/categories",
  "/api/proxy/api/store/search",
  "/api/proxy/api/store/skills",
  "/api/proxy/api/experts/templates",
  "/api/proxy/api/integrations/providers/system",
];

// Private routes nested under an allowed prefix need their own, longer
// disallow to keep winning the match.
const PRIVATE_API_PREFIXES = ["/api/proxy/api/store/skills/submissions"];

export default function robots(): MetadataRoute.Robots {
  const siteUrl = getSiteUrl();
  const allow = ["/", ...PUBLIC_API_PREFIXES];
  const disallow = Array.from(
    new Set([
      ...PRIVATE_PREFIXES,
      ...PRIVATE_API_PREFIXES,
      ...PROTECTED_PAGES,
      ...ADMIN_PAGES,
    ]),
  );

  return {
    rules: { userAgent: "*", allow, disallow },
    sitemap: new URL("/sitemap.xml", siteUrl).toString(),
  };
}
