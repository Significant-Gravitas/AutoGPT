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

describe("MethodPanel", () => {
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
