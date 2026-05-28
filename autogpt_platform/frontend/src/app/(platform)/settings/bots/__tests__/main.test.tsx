import { describe, expect, test } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getDeletePlatformLinkingUnlinkADmUserLinkMockHandler401,
  getDeletePlatformLinkingUnlinkAPlatformServerMockHandler,
  getListBotPlatformsMockHandler,
  getListBotPlatformsMockHandler401,
} from "@/app/api/__generated__/endpoints/platform-linking/platform-linking.msw";
import type { BotPlatformInfo } from "@/app/api/__generated__/models/botPlatformInfo";

import SettingsBotsPage from "../page";

function discordPlatform(
  overrides: Partial<BotPlatformInfo> = {},
): BotPlatformInfo {
  return {
    platform: "DISCORD",
    display_name: "Discord",
    icon: "discord.png",
    add_bot_url: "https://discord.com/oauth2/authorize?client_id=123",
    dm_link: undefined,
    server_links: [],
    ...overrides,
  };
}

describe("SettingsBotsPage", () => {
  test("renders the header and the Discord card with an Add bot button", async () => {
    server.use(getListBotPlatformsMockHandler([discordPlatform()]));

    render(<SettingsBotsPage />);

    expect(
      await screen.findByRole("heading", { name: /^bots$/i }),
    ).toBeDefined();
    expect(
      await screen.findByRole("heading", { name: /discord/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: /add bot to discord/i }),
    ).toBeDefined();
  });

  test("shows the 'no bots enabled' empty state when no platforms are configured", async () => {
    server.use(getListBotPlatformsMockHandler([]));

    render(<SettingsBotsPage />);

    expect(await screen.findByText(/no bots enabled/i)).toBeDefined();
  });

  test("renders an error card on 401 instead of the platform list", async () => {
    server.use(getListBotPlatformsMockHandler401());

    render(<SettingsBotsPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
  });

  test("shows the DM-not-linked tile and an unlinked indicator when no DM link", async () => {
    server.use(getListBotPlatformsMockHandler([discordPlatform()]));

    render(<SettingsBotsPage />);

    expect(
      await screen.findByText(/dm the bot on discord to link/i),
    ).toBeDefined();
    expect(screen.getByText(/not linked/i)).toBeDefined();
  });

  test("shows the DM username and an Unlink button when DM link exists", async () => {
    server.use(
      getListBotPlatformsMockHandler([
        discordPlatform({
          dm_link: {
            id: "dm-1",
            platform: "DISCORD",
            platform_user_id: "u-1",
            platform_username: "bently",
            linked_at: new Date("2024-01-01T00:00:00Z"),
          },
        }),
      ]),
    );

    render(<SettingsBotsPage />);

    expect(await screen.findByText("bently")).toBeDefined();
    expect(
      screen.getByRole("button", { name: /unlink dm on discord/i }),
    ).toBeDefined();
  });

  test("shows a 'Bot not in server' indicator when the server name is missing", async () => {
    server.use(
      getListBotPlatformsMockHandler([
        discordPlatform({
          server_links: [
            {
              id: "srv-orphan",
              platform: "DISCORD",
              platform_server_id: "1126875755960336515",
              owner_platform_user_id: "u-1",
              server_name: null,
              linked_at: new Date("2024-01-01T00:00:00Z"),
            },
          ],
        }),
      ]),
    );

    render(<SettingsBotsPage />);

    expect(await screen.findByText("1126875755960336515")).toBeDefined();
    expect(screen.getByText(/bot not in server/i)).toBeDefined();
  });

  test("clicking unlink on a linked server fires the API and the row is gone after refetch", async () => {
    server.use(
      getListBotPlatformsMockHandler([
        discordPlatform({
          server_links: [
            {
              id: "srv-1",
              platform: "DISCORD",
              platform_server_id: "111",
              owner_platform_user_id: "u-1",
              server_name: "AutoGPT HQ",
              linked_at: new Date("2024-01-01T00:00:00Z"),
            },
          ],
        }),
      ]),
      getDeletePlatformLinkingUnlinkAPlatformServerMockHandler({
        success: true,
      }),
    );

    render(<SettingsBotsPage />);

    const unlinkBtn = await screen.findByRole("button", {
      name: /unlink autogpt hq/i,
    });

    server.use(getListBotPlatformsMockHandler([discordPlatform()]));
    fireEvent.click(unlinkBtn);

    await waitFor(() => {
      expect(screen.queryByText("AutoGPT HQ")).toBeNull();
    });
  });

  test("a 401 on DM unlink keeps the row in place", async () => {
    server.use(
      getListBotPlatformsMockHandler([
        discordPlatform({
          dm_link: {
            id: "dm-1",
            platform: "DISCORD",
            platform_user_id: "u-1",
            platform_username: "bently",
            linked_at: new Date("2024-01-01T00:00:00Z"),
          },
        }),
      ]),
      getDeletePlatformLinkingUnlinkADmUserLinkMockHandler401(),
    );

    render(<SettingsBotsPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: /unlink dm on discord/i }),
    );

    // Swap the GET handler so a refetch would drop the DM row. If the failed
    // mutation wrongly invalidated platforms, "bently" would vanish. Asserting
    // it stays proves no refetch fired.
    server.use(getListBotPlatformsMockHandler([discordPlatform()]));
    await waitFor(() => {
      expect(screen.getByText("bently")).toBeDefined();
    });
  });
});
