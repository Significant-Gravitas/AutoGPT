import { NextRequest, NextResponse } from "next/server";

// Serves the changelog hero images (from our `gitbook` branch) through our own
// cached origin, so browsers never hotlink raw.githubusercontent.com.
const RAW_ASSETS_BASE =
  "https://raw.githubusercontent.com/Significant-Gravitas/AutoGPT/gitbook/docs/platform/.gitbook/assets/";
const FILE_PATTERN = /^[a-zA-Z0-9._-]+\.(png|jpe?g|gif|webp|svg)$/;

export async function GET(request: NextRequest) {
  const file = request.nextUrl.searchParams.get("file");

  if (!file || !FILE_PATTERN.test(file)) {
    return NextResponse.json({ error: "Invalid file" }, { status: 400 });
  }

  const target = new URL(file, RAW_ASSETS_BASE);
  // Defense-in-depth against SSRF: only the fixed assets directory is reachable.
  if (!target.href.startsWith(RAW_ASSETS_BASE)) {
    return NextResponse.json({ error: "Invalid path" }, { status: 400 });
  }

  try {
    const upstream = await fetch(target.href, {
      // Images rarely change; cache them for a day.
      next: { revalidate: 86400 },
    });

    if (!upstream.ok) {
      return NextResponse.json(
        { error: "Upstream image fetch failed" },
        { status: 502 },
      );
    }

    const body = await upstream.arrayBuffer();
    return new NextResponse(body, {
      status: 200,
      headers: {
        "Content-Type": upstream.headers.get("content-type") ?? "image/png",
        "Cache-Control": "public, max-age=86400, s-maxage=604800",
      },
    });
  } catch {
    return NextResponse.json(
      { error: "Upstream image fetch failed" },
      { status: 502 },
    );
  }
}
