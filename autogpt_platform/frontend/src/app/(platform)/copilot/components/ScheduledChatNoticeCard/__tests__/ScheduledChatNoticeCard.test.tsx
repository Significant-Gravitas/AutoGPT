import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import { getResumeScheduledChatMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { QueryClient } from "@tanstack/react-query";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ScheduledChatNoticeCard } from "../ScheduledChatNoticeCard";
import type { ScheduledChatNotice } from "../helpers";

const SKIPPED: ScheduledChatNotice = {
  reason: "usage",
  headline: "Skipped: daily usage limit reached",
  body: "“Briefing” didn't run. Scheduled chats run again when the limit resets at 00:00 UTC.",
  scheduleKey: "routine-1",
  actions: [{ kind: "link", label: "Upgrade", href: "/settings/billing" }],
};

const PAUSED: ScheduledChatNotice = {
  reason: "connection",
  headline: "Paused: “Briefing” can't use Twitter",
  body: "Its last 3 runs failed because the Twitter connection is missing or was refused.",
  scheduleKey: "routine-1",
  actions: [
    { kind: "reconnect", label: "Reconnect Twitter", provider: "twitter" },
    { kind: "resume", label: "Resume" },
  ],
};

afterEach(() => {
  server.resetHandlers();
});

describe("ScheduledChatNoticeCard", () => {
  it("says why the run was skipped and links to the fix", () => {
    render(<ScheduledChatNoticeCard notice={SKIPPED} />);

    expect(
      screen.getByText("Skipped: daily usage limit reached"),
    ).toBeDefined();
    expect(screen.getByText(/didn't run/)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Upgrade" }).getAttribute("href"),
    ).toBe("/settings/billing");
  });

  it("resumes the paused chat it is about", async () => {
    let resumedKey: string | null = null;
    server.use(
      getResumeScheduledChatMockHandler(({ params }) => {
        resumedKey = String(params.scheduleKey);
        return { resumed: true };
      }),
    );
    render(<ScheduledChatNoticeCard notice={PAUSED} sessionID="session-1" />);

    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() => expect(resumedKey).toBe("routine-1"));
    expect(
      await screen.findByRole("button", { name: "Resumed" }),
    ).toHaveProperty("disabled", true);
  });

  it("refreshes Home, which lists the paused chat, once it resumes", async () => {
    const invalidate = vi.spyOn(QueryClient.prototype, "invalidateQueries");
    server.use(getResumeScheduledChatMockHandler(() => ({ resumed: true })));
    render(<ScheduledChatNoticeCard notice={PAUSED} sessionID="session-1" />);

    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    await waitFor(() =>
      expect(invalidate).toHaveBeenCalledWith({
        queryKey: getGetHomeDashboardQueryKey(),
      }),
    );
    invalidate.mockRestore();
  });

  it("treats a chat that is no longer paused as resumed", async () => {
    server.use(
      http.post("*/api/schedules/followups/:scheduleKey/resume", () =>
        HttpResponse.json(
          { detail: "This scheduled chat is not paused." },
          { status: 404 },
        ),
      ),
    );
    render(<ScheduledChatNoticeCard notice={PAUSED} />);

    fireEvent.click(screen.getByRole("button", { name: "Resume" }));

    expect(
      await screen.findByRole("button", { name: "Resumed" }),
    ).toBeDefined();
  });

  it("opens the connect dialog for the provider that failed", async () => {
    server.use(
      getGetV1ListProvidersMockHandler([]),
      getGetV1ListCredentialsMockHandler([]),
    );
    render(<ScheduledChatNoticeCard notice={PAUSED} />);

    fireEvent.click(screen.getByRole("button", { name: "Reconnect Twitter" }));

    expect(await screen.findByRole("dialog")).toBeDefined();
  });
});
