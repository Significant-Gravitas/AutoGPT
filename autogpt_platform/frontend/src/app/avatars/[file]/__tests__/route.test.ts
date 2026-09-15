import { notionConfigForLegacyUrl } from "@/components/molecules/NotionAvatar/helpers";
import { notionAvatarUrlFor } from "@/components/molecules/NotionAvatar/helpers";
import { describe, expect, it } from "vitest";
import { GET } from "../route";

function get(file: string) {
  return GET(new Request("http://localhost/avatars/x"), {
    params: Promise.resolve({ file }),
  });
}

describe("GET /avatars/[file]", () => {
  it("redirects a legacy avatar permanently to its Notion face", async () => {
    const response = await get("round.sky.glasses.svg");

    expect(response.status).toBe(308);
    const target = response.headers.get("Location");
    expect(target).toContain("/avatars/notion/");
    // The same face the component resolves for that URL, so an expert does not
    // change appearance depending on which one drew them.
    const config = notionConfigForLegacyUrl("/avatars/round.sky.glasses.svg");
    expect(target).toContain(notionAvatarUrlFor(config!));
  });

  it("keeps the colour the expert was raised with", async () => {
    const response = await get("round.sky.glasses.svg");

    expect(response.headers.get("Location")).toContain(".sky.svg");
  });

  it("sends two different legacy shapes to two different faces", async () => {
    const [a, b] = await Promise.all([
      get("round.sky.glasses.svg"),
      get("bean.sky.crown.svg"),
    ]);

    expect(a.headers.get("Location")).not.toBe(b.headers.get("Location"));
  });

  it("404s on a file that is not an avatar spec", async () => {
    expect((await get("nope.svg")).status).toBe(404);
  });
});
