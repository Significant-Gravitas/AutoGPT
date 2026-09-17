import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";
import { isRenderableImageUrl } from "./next-image";

/** The hostnames next.config.mjs allows, read from the config's source, so a
 *  host added there but not to the helper fails this test. */
function configuredHosts(): string[] {
  const source = readFileSync(join(process.cwd(), "next.config.mjs"), "utf8");
  const domainsBlock = /domains:\s*\[([^\]]*)\]/.exec(source)?.[1] ?? "";
  const domains = [...domainsBlock.matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  const patterns = [...source.matchAll(/hostname:\s*"([^"]+)"/g)].map(
    (m) => m[1],
  );
  return Array.from(new Set([...domains, ...patterns]));
}

describe("isRenderableImageUrl", () => {
  test("accepts every host next.config.mjs configures", () => {
    const hosts = configuredHosts();
    expect(hosts.length).toBeGreaterThan(0);
    for (const host of hosts) {
      expect(isRenderableImageUrl(`https://${host}/agent.png`)).toBe(true);
    }
  });

  test("rejects a host next/image would throw on", () => {
    expect(isRenderableImageUrl("https://evil.example.org/agent.png")).toBe(
      false,
    );
  });

  test("rejects nothing to render and anything unparseable", () => {
    expect(isRenderableImageUrl(null)).toBe(false);
    expect(isRenderableImageUrl(undefined)).toBe(false);
    expect(isRenderableImageUrl("")).toBe(false);
    expect(isRenderableImageUrl("not a url")).toBe(false);
  });

  test("accepts an app-relative path", () => {
    expect(isRenderableImageUrl("/images/agent.png")).toBe(true);
  });
});
