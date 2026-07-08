import { describe, expect, it } from "vitest";

import {
  getGetV2BotMessageTimeseriesMockHandler200,
  getGetV2BotServerCountTimeseriesShardingCurveMockHandler200,
  getGetV2BotServerRosterMockHandler200,
  getGetV2BotUsageSummaryMockHandler200,
  getGetV2CommandUsageBreakdownMockHandler200,
  getGetV2TopServersByActivityMockHandler200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";

import { BotsContent } from "../components/BotsContent";

function mockAllEndpoints() {
  server.use(
    getGetV2BotUsageSummaryMockHandler200({
      platform: "DISCORD",
      window_days: 30,
      live_servers: 42,
      messages_received: 1000,
      replies_sent: 950,
      commands_used: 30,
      stream_errors: 5,
      avg_reply_ms: 1500,
      error_rate: 0.005,
    }),
    getGetV2BotMessageTimeseriesMockHandler200([
      {
        date: new Date("2026-06-01T00:00:00Z"),
        messages: 10,
        replies: 9,
        errors: 1,
      },
    ]),
    getGetV2BotServerCountTimeseriesShardingCurveMockHandler200([
      { date: new Date("2026-06-01T00:00:00Z"), server_count: 42 },
    ]),
    getGetV2TopServersByActivityMockHandler200([
      {
        platform: "DISCORD",
        server_id: "1",
        name: "Cool Guild",
        messages: 500,
        commands: 10,
      },
      {
        platform: "SLACK",
        server_id: "T1",
        name: "Cool Workspace",
        messages: 300,
        commands: 4,
      },
    ]),
    getGetV2CommandUsageBreakdownMockHandler200([
      { platform: "DISCORD", command: "setup", uses: 12 },
    ]),
    getGetV2BotServerRosterMockHandler200([
      {
        platform: "DISCORD",
        server_id: "1",
        name: "Cool Guild",
        joined_at: new Date("2026-05-01T00:00:00Z"),
        left_at: null,
        active: true,
      },
    ]),
  );
}

describe("BotsContent", () => {
  it("renders the live server count and headline metrics", async () => {
    mockAllEndpoints();
    render(<BotsContent />);

    expect(await screen.findByText("Live servers")).toBeDefined();
    expect(await screen.findByText("42")).toBeDefined();
    expect(await screen.findByText("0.5%")).toBeDefined();
  });

  it("renders server activity, command usage and roster rows", async () => {
    mockAllEndpoints();
    render(<BotsContent />);

    expect(await screen.findAllByText("Cool Guild")).toHaveLength(2);
    expect(await screen.findByText("/setup")).toBeDefined();
    expect(await screen.findByText("Active")).toBeDefined();
  });

  it("labels each row with its platform on the All platforms view", async () => {
    mockAllEndpoints();
    render(<BotsContent />);

    // Discord appears in all three tables (top servers, command usage, roster);
    // Slack only in the top-servers table — proving mixed rows are disambiguated.
    expect(
      (await screen.findAllByText("Discord")).length,
    ).toBeGreaterThanOrEqual(3);
    expect(await screen.findByText("Slack")).toBeDefined();
  });

  it("falls back to the raw platform name for an unknown platform", async () => {
    mockAllEndpoints();
    server.use(
      getGetV2TopServersByActivityMockHandler200([
        {
          platform: "TELEGRAM",
          server_id: "tg1",
          name: "Mystery",
          messages: 1,
          commands: 0,
        },
      ]),
    );
    render(<BotsContent />);

    // No label entry exists for TELEGRAM, so the badge shows the raw key.
    expect(await screen.findByText("TELEGRAM")).toBeDefined();
  });
});
