import { ADMIN_PAGES, PROTECTED_PAGES } from "@/lib/auth/helpers";
import { getSiteUrl } from "@/lib/metadata";
import type { MetadataRoute } from "next";

const PRIVATE_PREFIXES = ["/api", "/auth"];

export default function robots(): MetadataRoute.Robots {
  const siteUrl = getSiteUrl();
  const disallow = Array.from(
    new Set([...PRIVATE_PREFIXES, ...PROTECTED_PAGES, ...ADMIN_PAGES]),
  );

  return {
    rules: { userAgent: "*", allow: "/", disallow },
    sitemap: new URL("/sitemap.xml", siteUrl).toString(),
  };
}
