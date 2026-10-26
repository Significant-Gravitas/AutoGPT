import { NextResponse } from "next/server";

const PUBLISHED_CHANGELOG_INDEX =
  "https://agpt.co/docs/platform/changelog/changelog.md";

export async function GET() {
  try {
    const upstream = await fetch(PUBLISHED_CHANGELOG_INDEX, {
      headers: { Accept: "text/plain, text/markdown, */*" },
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
