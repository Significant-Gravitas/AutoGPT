import { parseChangelogIndex } from "@/components/molecules/ChangelogPopup/helpers";
import { NextResponse } from "next/server";
import {
  CHANGELOG_CACHE_CONTROL,
  CHANGELOG_FETCH_INIT,
  fetchReleaseTitle,
  INDEX_FETCH_TIMEOUT_MS,
  LatestRelease,
  PUBLISHED_CHANGELOG_INDEX,
} from "../helpers";

const LATEST_RELEASE_COUNT = 3;

// The newest releases with their page titles, for the logged-out login page.
export async function GET() {
  try {
    const upstream = await fetch(PUBLISHED_CHANGELOG_INDEX, {
      ...CHANGELOG_FETCH_INIT,
      signal: AbortSignal.timeout(INDEX_FETCH_TIMEOUT_MS),
    });
    if (!upstream.ok) return upstreamFailed();

    const entries = parseChangelogIndex(await upstream.text()).slice(
      0,
      LATEST_RELEASE_COUNT,
    );
    const releases: LatestRelease[] = await Promise.all(
      entries.map(async (entry) => ({
        ...entry,
        title: await fetchReleaseTitle(entry.slug),
      })),
    );

    return NextResponse.json(
      { releases },
      { headers: { "Cache-Control": CHANGELOG_CACHE_CONTROL } },
    );
  } catch {
    return upstreamFailed();
  }
}

function upstreamFailed() {
  return NextResponse.json(
    { error: "Upstream changelog fetch failed" },
    { status: 502 },
  );
}
