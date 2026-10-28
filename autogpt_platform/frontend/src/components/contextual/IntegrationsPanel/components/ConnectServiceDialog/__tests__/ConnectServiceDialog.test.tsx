import { getGetV1ListProvidersMockHandler } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { ProviderMetadata } from "@/app/api/__generated__/models/providerMetadata";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test } from "vitest";

import { ConnectServiceDialog } from "../ConnectServiceDialog";

function provider(
  name: string,
  overrides: Partial<ProviderMetadata> = {},
): ProviderMetadata {
  return {
    name,
    description: "Service",
    supported_auth_types: ["oauth2"],
    service: name,
    ...overrides,
  };
}

const PROVIDERS = [
  provider("mcp_1inch", {
    display_name: "1inch",
    supported_auth_types: [],
    service: "1inch",
    service_name: "1inch",
    mcp_server: {
      documentation_url: "https://1inch.io",
      setup_instructions: "Sign in.",
      connection_mode: "hosted",
      auth_methods: ["oauth"],
    },
  }),
  provider("aave", { description: "Lending markets" }),
  provider("airtable"),
  provider("discord"),
  provider("github"),
  provider("hubspot"),
  provider("linear"),
  provider("openai"),
  provider("twitter"),
  provider("google", { description: "Gmail, Drive, Calendar, Sheets" }),
  provider("notion"),
  provider("slack", { supported_auth_types: [] }),
  provider("zerobounce"),
];

function renderDialog() {
  server.use(getGetV1ListProvidersMockHandler(PROVIDERS));
  render(<ConnectServiceDialog open onOpenChange={() => {}} />);
}

function buttonsIn(list: HTMLElement) {
  return within(list)
    .getAllByRole("button")
    .map((button) => button.textContent);
}

function startingWith(...names: string[]) {
  return names.map((name) => expect.stringMatching(new RegExp(`^${name}`)));
}

describe("ConnectServiceDialog", () => {
  test("opens on popular services, with the rest in popularity order below", async () => {
    renderDialog();

    const popular = await screen.findByRole("region", { name: "Popular" });
    expect(buttonsIn(popular)).toEqual(
      startingWith(
        "Google",
        "GitHub",
        "Notion",
        "Linear",
        "Discord",
        "X",
        "HubSpot",
        "Airtable",
      ),
    );

    const more = screen.getByRole("region", { name: "More services" });
    expect(buttonsIn(more)).toEqual(
      startingWith("OpenAI", "ZeroBounce", "1inch", "Aave", "Slack"),
    );
  });

  test("ranks popular matches first when searching", async () => {
    renderDialog();
    await screen.findByRole("region", { name: "Popular" });

    await userEvent.type(screen.getByLabelText("Search services"), "o");

    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Popular" })).toBeNull(),
    );
    const dialog = screen.getByRole("dialog");
    expect(buttonsIn(within(dialog).getByRole("list"))).toEqual(
      startingWith(
        "Google",
        "Notion",
        "Discord",
        "HubSpot",
        "OpenAI",
        "ZeroBounce",
      ),
    );
  });

  test("says so when no service matches the search", async () => {
    renderDialog();
    await screen.findByRole("region", { name: "Popular" });

    await userEvent.type(screen.getByLabelText("Search services"), "qwxz");

    expect(await screen.findByText('No services match "qwxz"')).toBeDefined();
    expect(screen.queryByRole("region", { name: "Popular" })).toBeNull();
  });

  test("a popular service opens its connect step", async () => {
    renderDialog();
    const popular = await screen.findByRole("region", { name: "Popular" });

    await userEvent.click(
      within(popular).getByRole("button", { name: /^Google/ }),
    );

    expect(
      await screen.findByRole("button", { name: "Back to services" }),
    ).toBeDefined();
    expect(
      screen.getByRole("heading", { name: "Google", level: 2 }),
    ).toBeDefined();
  });
});
