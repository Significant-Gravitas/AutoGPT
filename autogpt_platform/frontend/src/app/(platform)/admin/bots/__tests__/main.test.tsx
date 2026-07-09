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
      { server_id: "1", name: "Cool Guild", messages: 500, commands: 10 },
    ]),
    getGetV2CommandUsageBreakdownMockHandler200([
      { command: "setup", uses: 12 },
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
});
