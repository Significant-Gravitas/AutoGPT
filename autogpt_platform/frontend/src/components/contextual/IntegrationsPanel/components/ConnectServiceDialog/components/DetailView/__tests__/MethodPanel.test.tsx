import { getGetV2ListProviderModelTiersMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { ProviderTiers } from "@/app/api/__generated__/models/providerTiers";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";

import { AuthType, type ConnectableProvider } from "../../../helpers";
import { MethodPanel } from "../MethodPanel";
import { useOAuthConnect } from "../useOAuthConnect";

vi.mock("../useOAuthConnect", () => ({
  useOAuthConnect: vi.fn(() => ({ connect: vi.fn(), isPending: false })),
}));

const openaiProvider: ConnectableProvider = {
  id: "openai",
  name: "OpenAI",
  description: "OpenAI models via API key or your ChatGPT subscription",
  supportedAuthTypes: [AuthType.oauth2, AuthType.api_key],
  authProviderByType: { [AuthType.oauth2]: "codex" },
};

const githubProvider: ConnectableProvider = {
  id: "github",
  name: "GitHub",
  description: "GitHub, or your Copilot subscription",
  supportedAuthTypes: [AuthType.oauth2],
  authProviderByType: { [AuthType.oauth2]: "github_copilot" },
};

const CHATGPT: ProviderTiers = {
  provider_family: "openai",
  display_name: "ChatGPT",
  auth_provider: "codex",
  connect_button_label: "Sign in with ChatGPT",
  terms_company: "OpenAI",
  tiers: [
    { tier: "standard", label: "Balanced", display_model: "GPT-5.6 Terra" },
    { tier: "advanced", label: "Advanced", display_model: "GPT-5.6 Sol" },
  ],
} as ProviderTiers;

const COPILOT: ProviderTiers = {
  provider_family: "github",
  display_name: "GitHub Copilot",
  auth_provider: "github_copilot",
  connect_button_label: "Sign in with GitHub",
  terms_company: "GitHub",
  tiers: [],
} as ProviderTiers;

function mockProviderTiers(providers: ProviderTiers[]) {
  server.use(getGetV2ListProviderModelTiersMockHandler200({ providers }));
}

describe("MethodPanel", () => {
  test("names the models the plan gets you, from the catalog", async () => {
    // Mock 2 spells these out. They cannot come from the connections list --
    // the user has not connected yet, so ChatGPT is absent from it -- and
    // hardcoding them here would drift from the catalog that routes the turn.
    mockProviderTiers([CHATGPT]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(
      await screen.findByText(
        /GPT-5\.6 Terra \(Balanced\) and GPT-5\.6 Sol \(Advanced\)/,
      ),
    ).toBeDefined();
  });

  test("falls back to the general promise when the catalog names nothing", async () => {
    mockProviderTiers([{ ...CHATGPT, tiers: [] }]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(
      await screen.findByText(/models your ChatGPT subscription already/),
    ).toBeDefined();
  });

  test("uses ChatGPT branding while sending OAuth to the Codex backend provider", async () => {
    mockProviderTiers([CHATGPT]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(
      await screen.findByRole("button", { name: "Sign in with ChatGPT" }),
    ).toBeDefined();
    expect(screen.getByText(/chatgpt sign-in window/i)).toBeDefined();
    expect(vi.mocked(useOAuthConnect)).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "codex" }),
    );
  });

  test("answers what linking a ChatGPT plan means before the OAuth window", async () => {
    mockProviderTiers([CHATGPT]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(await screen.findByText("What it does.")).toBeDefined();
    expect(screen.getByText("What it costs.")).toBeDefined();
    expect(screen.getByText("What you get.")).toBeDefined();
    expect(screen.getByText("Stay in control.")).toBeDefined();
    expect(screen.getByText(/spend zero AutoGPT credits/i)).toBeDefined();
    expect(screen.getByText(/follow OpenAI's terms/i)).toBeDefined();
  });

  test("says the same four things about a second subscription provider", async () => {
    // The point of the abstraction: a provider the server describes gets the
    // whole panel -- explainer, button, terms line -- with no branch here.
    // If this ever needs a code change to pass, the copy went back to being
    // ChatGPT's rather than the provider's.
    mockProviderTiers([CHATGPT, COPILOT]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={githubProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(
      await screen.findByRole("button", { name: "Sign in with GitHub" }),
    ).toBeDefined();
    expect(screen.getByText("What it costs.")).toBeDefined();
    expect(
      screen.getByText(/your GitHub Copilot subscription's own limits/),
    ).toBeDefined();
    expect(screen.getByText(/follow GitHub's terms/i)).toBeDefined();
    // ChatGPT is in the same response; naming it here would mean the panel
    // matched on the family, or on nothing at all.
    expect(screen.queryByText(/ChatGPT/)).toBeNull();
    expect(vi.mocked(useOAuthConnect)).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "github_copilot" }),
    );
  });

  test("claims nothing it cannot back up", async () => {
    mockProviderTiers([{ ...CHATGPT, tiers: [] }]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    await screen.findByText("What it does.");
    // Model names are server-owned, and pause-and-resume at a provider limit
    // is not built yet — neither may be promised here.
    expect(screen.queryByText(/5\.6|Terra|Sol|Balanced|Advanced/)).toBeNull();
    expect(screen.queryByText(/the run pauses/i)).toBeNull();
  });

  test("does not show the subscription explainer for an unrelated provider", async () => {
    mockProviderTiers([CHATGPT]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={{
          id: "notion",
          name: "Notion",
          description: "Notion",
          supportedAuthTypes: [AuthType.oauth2],
        }}
        onSuccess={vi.fn()}
      />,
    );

    expect(await screen.findByText(/Notion sign-in window/)).toBeDefined();
    expect(screen.queryByText("What it costs.")).toBeNull();
    expect(screen.queryByText(/follow Notion's terms/i)).toBeNull();
  });
});
