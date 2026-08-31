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
});
