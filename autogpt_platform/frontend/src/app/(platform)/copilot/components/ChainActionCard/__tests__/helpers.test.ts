import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { describe, expect, it, vi } from "vitest";
import {
  formatInputsTitle,
  toConnectorRows,
  type ConnectorRequest,
} from "../helpers";

function connectorRequest(
  overrides: Partial<ConnectorRequest> = {},
): ConnectorRequest {
  return {
    id: "req-1",
    fields: [
      [
        "credentials",
        {
          credentials_provider: ["github"],
          credentials_types: ["api_key"],
        },
      ],
    ],
    selected: {},
    onChange: vi.fn(),
    ...overrides,
  };
}

describe("formatInputsTitle", () => {
  it("converts a block class name to spaced words without the Block suffix", () => {
    expect(formatInputsTitle("SendDiscordMessageBlock")).toBe(
      "Send Discord Message",
    );
  });

  it("converts underscores to spaces", () => {
    expect(formatInputsTitle("my_agent_name")).toBe("my agent name");
  });

  it("passes agent names with regular spacing through unchanged", () => {
    expect(formatInputsTitle("Market Research Agent")).toBe(
      "Market Research Agent",
    );
  });

  it("falls back to the original name when stripping leaves nothing", () => {
    expect(formatInputsTitle("Block")).toBe("Block");
  });
});

describe("toConnectorRows", () => {
  it("merges requests asking for the same provider into one row", () => {
    const first = connectorRequest({ id: "req-1" });
    const second = connectorRequest({ id: "req-2" });

    const rows = toConnectorRows([first, second], []);

    expect(rows).toHaveLength(1);
    expect(rows[0].provider).toBe("github");
  });

  it("fans a selection out to every request asking for the provider", () => {
    const first = connectorRequest({ id: "req-1" });
    const second = connectorRequest({ id: "req-2" });
    const credential = {
      id: "cred-1",
      provider: "github",
      type: "api_key" as const,
    };

    const [row] = toConnectorRows([first, second], []);
    row.select(credential);

    expect(first.onChange).toHaveBeenCalledWith("credentials", credential);
    expect(second.onChange).toHaveBeenCalledWith("credentials", credential);
  });

  it("drops fields whose provider cannot be resolved", () => {
    const request = connectorRequest({
      fields: [["credentials", { credentials_types: ["api_key"] }]],
    });

    expect(toConnectorRows([request], [])).toHaveLength(0);
  });

  it("prefers the provider metadata description over the schema description", () => {
    const request = connectorRequest({
      fields: [
        [
          "credentials",
          {
            credentials_provider: ["github"],
            description: "From the schema",
          },
        ],
      ],
    });
    const providers: ProviderMetadata[] = [
      { name: "github", description: "From the metadata" },
    ];

    const [row] = toConnectorRows([request], providers);

    expect(row.description).toBe("From the metadata");
    expect(row.displayName).toBe("GitHub");
  });

  it("falls back to the schema description, then to null", () => {
    const withSchemaDescription = connectorRequest({
      fields: [
        [
          "credentials",
          {
            credentials_provider: ["github"],
            description: "From the schema",
          },
        ],
      ],
    });
    expect(toConnectorRows([withSchemaDescription], [])[0].description).toBe(
      "From the schema",
    );

    const withoutDescription = connectorRequest();
    expect(toConnectorRows([withoutDescription], [])[0].description).toBeNull();
  });

  it("keeps the first selected credential for the provider", () => {
    const credential = {
      id: "cred-1",
      provider: "github",
      type: "api_key" as const,
    };
    const withSelection = connectorRequest({
      id: "req-1",
      selected: { credentials: credential },
    });
    const withoutSelection = connectorRequest({ id: "req-2" });

    const [row] = toConnectorRows([withSelection, withoutSelection], []);

    expect(row.selected).toEqual(credential);
  });

  it("formats unknown provider slugs into title case display names", () => {
    const request = connectorRequest({
      fields: [
        ["credentials", { credentials_provider: ["my_custom_service"] }],
      ],
    });

    expect(toConnectorRows([request], [])[0].displayName).toBe(
      "My Custom Service",
    );
  });
});
