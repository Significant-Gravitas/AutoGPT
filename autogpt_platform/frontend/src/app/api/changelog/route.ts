import { NextResponse } from "next/server";
import {
  CHANGELOG_CACHE_CONTROL,
  CHANGELOG_FETCH_INIT,
  PUBLISHED_CHANGELOG_INDEX,
} from "./helpers";

export async function GET() {
  try {
    const upstream = await fetch(
      PUBLISHED_CHANGELOG_INDEX,
      CHANGELOG_FETCH_INIT,
    );

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
        "Cache-Control": CHANGELOG_CACHE_CONTROL,
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Upstream changelog fetch failed" },
      { status: 502 },
    );
  }
}
