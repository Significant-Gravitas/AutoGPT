import { NextRequest, NextResponse } from "next/server";

// The docs site (GitBook, behind Cloudflare) 503s cross-origin browser fetches
// of the raw `.md`, so the changelog UI can't fetch it directly. This route
// proxies the fetch server-side (same-origin to the browser) and is locked to
// the changelog docs path — the only accepted input is a `[a-z0-9-]` slug — so
// it can't be used as an open proxy.
const DOCS_CHANGELOG_BASE = "https://agpt.co/docs/platform/changelog/changelog";
const SLUG_PATTERN = /^[a-z0-9-]+$/;

export async function GET(request: NextRequest) {
  const slug = request.nextUrl.searchParams.get("slug");

  if (slug !== null && !SLUG_PATTERN.test(slug)) {
    return NextResponse.json({ error: "Invalid slug" }, { status: 400 });
  }

  const target = slug
    ? `${DOCS_CHANGELOG_BASE}/${slug}.md`
    : `${DOCS_CHANGELOG_BASE}.md`;

  try {
    const upstream = await fetch(target, {
      headers: {
        "User-Agent": "AutoGPT-Platform-Changelog/1.0",
        Accept: "text/markdown, text/plain, */*",
      },
      // Cache the docs markdown so we don't hammer the docs site per view.
      next: { revalidate: 3600 },
    });

    if (!upstream.ok) {
      return NextResponse.json(
        { error: "Upstream changelog fetch failed" },
        { status: 502 },
      );
    }

    const markdown = await upstream.text();
    return new NextResponse(markdown, {
      status: 200,
      headers: {
        "Content-Type": "text/markdown; charset=utf-8",
        "Cache-Control": "public, max-age=300, s-maxage=3600",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Upstream changelog fetch failed" },
      { status: 502 },
    );
  }
}
