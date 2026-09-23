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

function mockProviderTiers(providers: ProviderTiers[]) {
  server.use(getGetV2ListProviderModelTiersMockHandler200({ providers }));
}

describe("MethodPanel", () => {
  test("names the models the plan gets you, from the catalog", async () => {
    // Mock 2 spells these out. They cannot come from the connections list --
    // the user has not connected yet, so ChatGPT is absent from it -- and
    // hardcoding them here would drift from the catalog that routes the turn.
    mockProviderTiers([
      {
        provider_family: "openai",
        display_name: "ChatGPT",
        tiers: [
          {
            tier: "standard",
            label: "Balanced",
            display_model: "GPT-5.6 Terra",
          },
          { tier: "advanced", label: "Advanced", display_model: "GPT-5.6 Sol" },
        ],
      } as ProviderTiers,
    ]);

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
    mockProviderTiers([]);

    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(
      await screen.findByText(/models your ChatGPT plan already includes/),
    ).toBeDefined();
  });

  test("uses ChatGPT branding while sending OAuth to the Codex backend provider", () => {
    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(screen.getByText(/chatgpt sign-in window/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Sign in with ChatGPT" }),
    ).toBeDefined();
    expect(vi.mocked(useOAuthConnect)).toHaveBeenCalledWith(
      expect.objectContaining({ provider: "codex" }),
    );
  });

  test("answers what linking a ChatGPT plan means before the OAuth window", () => {
    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    expect(screen.getByText("What it does.")).toBeDefined();
    expect(screen.getByText("What it costs.")).toBeDefined();
    expect(screen.getByText("What you get.")).toBeDefined();
    expect(screen.getByText("Stay in control.")).toBeDefined();
    expect(screen.getByText(/spend zero AutoGPT credits/i)).toBeDefined();
    expect(screen.getByText(/follow OpenAI's terms/i)).toBeDefined();
  });

  test("claims nothing it cannot back up", () => {
    render(
      <MethodPanel
        method={AuthType.oauth2}
        provider={openaiProvider}
        onSuccess={vi.fn()}
      />,
    );

    // Model names are server-owned, and pause-and-resume at a provider limit
    // is not built yet — neither may be promised here.
    expect(screen.queryByText(/5\.6|Terra|Sol|Balanced|Advanced/)).toBeNull();
    expect(screen.queryByText(/the run pauses/i)).toBeNull();
  });

  test("does not show the ChatGPT explainer for an unrelated provider", () => {
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

    expect(screen.queryByText("What it costs.")).toBeNull();
    expect(screen.queryByText(/follow Notion's terms/i)).toBeNull();
  });
});
