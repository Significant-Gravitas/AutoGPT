import { NextResponse } from "next/server";

// The changelog index lives on our `gitbook` branch. We proxy this single fixed
// URL server-side (cached) so browsers hit our own origin — not GitHub — and
// never treat raw.githubusercontent.com as a CDN. No user input is involved.
const RAW_CHANGELOG_INDEX =
  "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/gitbook/docs/platform/changelog/README.md";

export async function GET() {
  try {
    const upstream = await fetch(RAW_CHANGELOG_INDEX, {
      headers: { Accept: "text/plain, text/markdown, */*" },
      // Cache the index so we hit GitHub at most ~once an hour, not per view.
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
