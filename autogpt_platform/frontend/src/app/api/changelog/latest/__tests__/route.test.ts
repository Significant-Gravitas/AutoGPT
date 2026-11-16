import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { CHANGELOG_TIMEOUT_MS } from "@/app/(no-navbar)/login/components/LoginWhatsNew/helpers";
import { INDEX_FETCH_TIMEOUT_MS, RELEASE_PAGE_TIMEOUT_MS } from "../../helpers";
import { GET } from "../route";

const BASE = "https://agpt.co/docs/platform/changelog/changelog";

function row(slug: string, highlights = "Some news") {
  return `| [${slug}](/docs/platform/changelog/changelog/${slug}.md) | ${highlights} |`;
}

function page(title: string) {
  return `> For the complete documentation index, see [llms.txt](https://agpt.co/docs/llms.txt).\n\n# ${title}\n\n*October 8 – October 14, 2026*\n\n## A section\n`;
}

// An upstream that never answers, only giving up when the request is aborted.
const STALL = Symbol("stall");

function serve(pages: Record<string, Response | Error | typeof STALL>) {
  return vi.spyOn(global, "fetch").mockImplementation(async (input, init) => {
    const reply = pages[String(input)];
    if (reply === undefined) throw new Error(`Unexpected fetch ${input}`);
    if (reply === STALL) {
      return new Promise<Response>((_, reject) =>
        init?.signal?.addEventListener("abort", () =>
          reject(init.signal?.reason),
        ),
      );
    }
    if (reply instanceof Error) throw reply;
    return reply.clone();
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("GET /api/changelog/latest", () => {
  it("returns the three newest releases with their page titles", async () => {
    const fetchSpy = serve({
      [`${BASE}.md`]: new Response(
        [
          row("c-2026", "Wallet, trial"),
          row("b-2026"),
          row("a-2026"),
          row("old-2026"),
        ].join("\n"),
      ),
      [`${BASE}/c-2026.md`]: new Response(page("A clearer wallet")),
      [`${BASE}/b-2026.md`]: new Response(page("A bigger roster")),
      [`${BASE}/a-2026.md`]: new Response(page("Meet your Experts")),
    });

    const res = await GET();

    expect(res.status).toBe(200);
    expect(res.headers.get("Cache-Control")).toBe(
      "public, max-age=300, s-maxage=3600",
    );
    expect(await res.json()).toEqual({
      releases: [
        {
          slug: "c-2026",
          dateRange: "c-2026",
          highlights: "Wallet, trial",
          url: `${BASE}/c-2026`,
          title: "A clearer wallet",
        },
        expect.objectContaining({ slug: "b-2026", title: "A bigger roster" }),
        expect.objectContaining({ slug: "a-2026", title: "Meet your Experts" }),
      ],
    });
    expect(fetchSpy).toHaveBeenCalledTimes(4);
    expect(fetchSpy).toHaveBeenCalledWith(
      `${BASE}/c-2026.md`,
      expect.objectContaining({ next: { revalidate: 3600 } }),
    );
  });

  it("leaves half the login page's timeout for the network", () => {
    expect(
      INDEX_FETCH_TIMEOUT_MS + RELEASE_PAGE_TIMEOUT_MS,
    ).toBeLessThanOrEqual(CHANGELOG_TIMEOUT_MS / 2);
  });

  describe("when upstream stalls", () => {
    beforeEach(() => {
      vi.useFakeTimers();
      // AbortSignal.timeout runs on Node's own clock; drive it from the fake one.
      vi.spyOn(AbortSignal, "timeout").mockImplementation((ms) => {
        const controller = new AbortController();
        setTimeout(
          () => controller.abort(new DOMException("timed out", "TimeoutError")),
          ms,
        );
        return controller.signal;
      });
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("gives up on an index that never answers", async () => {
      serve({ [`${BASE}.md`]: STALL });

      const pending = GET();
      await vi.advanceTimersByTimeAsync(INDEX_FETCH_TIMEOUT_MS);
      const res = await pending;

      expect(res.status).toBe(502);
    });

    it("drops only the title of a page that never answers", async () => {
      serve({
        [`${BASE}.md`]: new Response(
          [row("c-2026"), row("b-2026"), row("a-2026")].join("\n"),
        ),
        [`${BASE}/c-2026.md`]: STALL,
        [`${BASE}/b-2026.md`]: new Response(page("A bigger roster")),
        [`${BASE}/a-2026.md`]: new Response(page("Meet your Experts")),
      });

      const pending = GET();
      await vi.advanceTimersByTimeAsync(RELEASE_PAGE_TIMEOUT_MS);
      const res = await pending;

      expect(res.status).toBe(200);
      const { releases } = await res.json();
      expect(
        releases.map((release: { title: string | null }) => release.title),
      ).toEqual([null, "A bigger roster", "Meet your Experts"]);
    });
  });

  it("leaves out only the title of a release page that can't be read", async () => {
    serve({
      [`${BASE}.md`]: new Response(
        [row("c-2026"), row("b-2026"), row("a-2026")].join("\n"),
      ),
      [`${BASE}/c-2026.md`]: new Response("missing", { status: 404 }),
      [`${BASE}/b-2026.md`]: new Error("network"),
      [`${BASE}/a-2026.md`]: new Response("No heading here, ## or #nope"),
    });

    const res = await GET();

    expect(res.status).toBe(200);
    const { releases } = await res.json();
    expect(
      releases.map((release: { title: unknown }) => release.title),
    ).toEqual([null, null, null]);
  });

  it("never builds a release page URL from a malformed slug", async () => {
    const fetchSpy = serve({
      [`${BASE}.md`]: new Response([row("a--b"), row("-x")].join("\n")),
    });

    const res = await GET();

    expect(res.status).toBe(200);
    const { releases } = await res.json();
    expect(releases).toEqual([
      expect.objectContaining({ slug: "a--b", title: null }),
      expect.objectContaining({ slug: "-x", title: null }),
    ]);
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("returns no releases when the index lists none", async () => {
    serve({ [`${BASE}.md`]: new Response("# What's New\n\nNothing yet.") });

    expect(await (await GET()).json()).toEqual({ releases: [] });
  });

  it.each([
    ["the index is unavailable", new Response("nope", { status: 503 })],
    ["the index fetch throws", new Error("network")],
  ])("returns 502 when %s", async (_case, reply) => {
    serve({ [`${BASE}.md`]: reply });

    expect((await GET()).status).toBe(502);
  });
});
