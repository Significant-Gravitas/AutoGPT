import { NextResponse } from "next/server";

// The docs site (GitBook, behind Cloudflare) 503s cross-origin browser fetches
// of the raw `.md`, so the changelog UI can't fetch the index directly. This
// route proxies that single fixed URL server-side (same-origin to the browser).
const DOCS_CHANGELOG_INDEX =
  "https://agpt.co/docs/platform/changelog/changelog.md";

export async function GET() {
  try {
    const upstream = await fetch(DOCS_CHANGELOG_INDEX, {
      headers: {
        "User-Agent": "AutoGPT-Platform-Changelog/1.0",
        Accept: "text/markdown, text/plain, */*",
      },
      // Cache the docs index so we don't hammer the docs site per view.
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
