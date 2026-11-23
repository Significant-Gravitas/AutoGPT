import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, test } from "vitest";
import { formatProviderName } from "../src/components/contextual/IntegrationsPanel/helpers";
import { integrationIconSrc } from "../src/components/molecules/IntegrationLogo/helpers";

interface CatalogEntry {
  name: string;
  display_name: string;
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

  // A skill names the MCP service it needs by the entry's name (#17313).
  test.each(catalog)("$name, named as a provider, shows that logo", (entry) => {
    expect(integrationIconSrc(entry.name)).toBe(
      integrationIconSrc(entry.mcp_server.icon_id!),
    );
    expectPNG(entry.name);
  });
});

describe("MCP catalog names", () => {
  // Every catalog entry, named as a provider, reads as its display_name
  // (#17336).
  test.each(catalog)(
    "$name, named as a provider, reads $display_name",
    (entry) => {
      expect(
        formatProviderName(entry.name),
        "a display_name the name after mcp_ doesn't format to belongs in " +
          "MCP_DISPLAY_NAMES (src/components/contextual/IntegrationsPanel/helpers.ts)",
      ).toBe(entry.display_name);
    },
  );
});
