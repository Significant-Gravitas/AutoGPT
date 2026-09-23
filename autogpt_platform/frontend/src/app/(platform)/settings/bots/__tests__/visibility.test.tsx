import { describe, expect, test, vi } from "vitest";

import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListBotPlatformsMockHandler } from "@/app/api/__generated__/endpoints/platform-linking/platform-linking.msw";

import SettingsBotsPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: vi.fn(() => ({ slack: false })),
  };
});

function platform(name: string) {
  return {
    platform: name.toUpperCase(),
    display_name: name,
    icon: `${name.toLowerCase()}.png`,
    add_bot_url: null,
    dm_link: undefined,
    server_links: [],
  };
}

describe("SettingsBotsPage platform visibility flag", () => {
  test("hides platforms the copilot-bot-platforms flag marks false", async () => {
    server.use(
      getListBotPlatformsMockHandler([platform("Discord"), platform("Slack")]),
    );

    render(<SettingsBotsPage />);

    expect(
      await screen.findByRole("heading", { name: /discord/i }),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { name: /slack/i })).toBeNull();
  });
});
