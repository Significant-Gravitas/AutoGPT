import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { integrationIconSrc } from "../src/components/molecules/IntegrationLogo/helpers";

interface CatalogEntry {
  name: string;
  mcp_server: { icon_id?: string };
}

const catalog: CatalogEntry[] = JSON.parse(
  readFileSync(
    resolve("../backend/backend/integrations/mcp_catalog.json"),
    "utf8",
  ),
);

function expectPNG(provider: string) {
  const src = integrationIconSrc(provider);
  expect(src).toBeTruthy();
  const bytes = readFileSync(resolve("public", `.${src}`));
  expect([...bytes.subarray(0, 8)]).toEqual([137, 80, 78, 71, 13, 10, 26, 10]);
  expect(bytes.readUInt32BE(16)).toBeGreaterThan(0);
  expect(bytes.readUInt32BE(20)).toBeGreaterThan(0);
}

describe("integration logo assets", () => {
  test.each(catalog)("$name has a bundled brand logo", (entry) => {
    expect(entry.mcp_server.icon_id).toBeTruthy();
    expectPNG(entry.mcp_server.icon_id!);
  });
});
