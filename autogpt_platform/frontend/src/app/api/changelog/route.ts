import { NextRequest, NextResponse } from "next/server";

// The changelog source lives on our `gitbook` branch. We proxy it server-side
// (cached) so browsers hit our own origin — not GitHub — and never treat
// raw.githubusercontent.com as a CDN. Locked to the changelog dir + a validated
// `[a-z0-9-]` slug so it can't be used as an open proxy.
const RAW_CHANGELOG_BASE =
  "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/gitbook/docs/platform/changelog";
const SLUG_PATTERN = /^[a-z0-9-]+$/;

export async function GET(request: NextRequest) {
  const slug = request.nextUrl.searchParams.get("slug");

  if (slug !== null && !SLUG_PATTERN.test(slug)) {
    return NextResponse.json({ error: "Invalid slug" }, { status: 400 });
  }

  const target = slug
    ? `${RAW_CHANGELOG_BASE}/${slug}.md`
    : `${RAW_CHANGELOG_BASE}/README.md`;

  try {
    const upstream = await fetch(target, {
      headers: { Accept: "text/plain, text/markdown, */*" },
      // Cache the markdown so we hit GitHub at most ~once an hour, not per view.
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
