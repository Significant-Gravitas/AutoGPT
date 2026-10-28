import type { ChangelogEntry } from "@/components/molecules/ChangelogPopup/helpers";

const PUBLISHED_CHANGELOG = "https://agpt.co/docs/platform/changelog/changelog";

export const PUBLISHED_CHANGELOG_INDEX = `${PUBLISHED_CHANGELOG}.md`;

export const CHANGELOG_FETCH_INIT = {
  headers: { Accept: "text/plain, text/markdown, */*" },
  next: { revalidate: 3600 },
};

export const CHANGELOG_CACHE_CONTROL = "public, max-age=300, s-maxage=3600";

// The latest-releases route fetches the index, then the pages in parallel.
// Together these take at most half the login page's own timeout (a test
// holds it), so the route answers well before the page gives up on it.
export const INDEX_FETCH_TIMEOUT_MS = 2000;

// A slow release page costs only its title, never the whole list.
export const RELEASE_PAGE_TIMEOUT_MS = 2000;

// Release page URLs are built from slugs, so only plain kebab-case passes.
const SLUG_PATTERN = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

// The page's first top-level heading: "# A clearer wallet, …".
const TITLE_PATTERN = /^#[ \t]+(.+?)[ \t#]*$/m;

export interface LatestRelease extends ChangelogEntry {
  // The release page's own title, or null when it couldn't be read.
  title: string | null;
}

export function isSafeSlug(slug: string) {
  return slug.length <= 100 && SLUG_PATTERN.test(slug);
}

export function getReleaseTitle(markdown: string): string | null {
  return TITLE_PATTERN.exec(markdown)?.[1].trim() || null;
}

// Resolves to null on any failure: the release then heads with a highlight.
export async function fetchReleaseTitle(slug: string): Promise<string | null> {
  if (!isSafeSlug(slug)) return null;
  try {
    const page = await fetch(`${PUBLISHED_CHANGELOG}/${slug}.md`, {
      ...CHANGELOG_FETCH_INIT,
      signal: AbortSignal.timeout(RELEASE_PAGE_TIMEOUT_MS),
    });
    if (!page.ok) return null;
    return getReleaseTitle(await page.text());
  } catch {
    return null;
  }
}
