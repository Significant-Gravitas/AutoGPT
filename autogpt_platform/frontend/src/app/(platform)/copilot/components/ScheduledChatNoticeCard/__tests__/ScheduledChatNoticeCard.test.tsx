import {
  getGetV1ListCredentialsMockHandler,
  getGetV1ListProvidersMockHandler,
} from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { getGetHomeDashboardQueryKey } from "@/app/api/__generated__/endpoints/home/home";
import {
  getGetScheduledChatCadenceMockHandler,
  getResumeScheduledChatMockHandler,
  getSetScheduledChatCadenceMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { ScheduledChatCadence } from "@/app/api/__generated__/models/scheduledChatCadence";
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

const OVER_LIMIT: ScheduledChatNotice = {
  ...SKIPPED,
  actions: [
    { kind: "link", label: "Upgrade", href: "/settings/billing" },
    { kind: "run_less_often", label: "Run less often" },
  ],
};

const CADENCE: ScheduledChatCadence = {
  schedule_key: "routine-1",
  title: "Discord monitor",
  current: "Every minute",
  timezone: "America/Chicago",
  options: [
    { cron: "*/15 * * * *", label: "Every 15 minutes" },
    { cron: "0 * * * *", label: "Every hour" },
    { cron: "0 9 * * *", label: "Every day at 09:00" },
    { cron: "0 9 * * mon-fri", label: "Weekdays at 09:00" },
  ],
};

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: (...args: unknown[]) => toastMock(...args) };
});

afterEach(() => {
  server.resetHandlers();
  toastMock.mockClear();
});

async function openRunLessOften(notice: ScheduledChatNotice = OVER_LIMIT) {
  render(<ScheduledChatNoticeCard notice={notice} sessionID="session-1" />);
  fireEvent.click(screen.getByRole("button", { name: "Run less often" }));
}

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

  it("runs the chat less often, at the cadence picked", async () => {
    let posted: unknown = null;
    server.use(
      getGetScheduledChatCadenceMockHandler(CADENCE),
      getSetScheduledChatCadenceMockHandler(async ({ params, request }) => {
        posted = { key: params.scheduleKey, body: await request.json() };
        return { ...CADENCE, current: "5 times a week", options: [] };
      }),
    );
    await openRunLessOften();

    expect(
      await screen.findByText("Run “Discord monitor” less often"),
    ).toBeDefined();
    expect(screen.getByText(/It runs every minute now/)).toBeDefined();
    expect(screen.getByRole("button", { name: "Save" })).toHaveProperty(
      "disabled",
      true,
    );

    fireEvent.click(screen.getByRole("radio", { name: "Weekdays at 09:00" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(posted).toEqual({
        key: "routine-1",
        body: { cron: "0 9 * * mon-fri" },
      }),
    );
    // The choice in the owner's words, not "5 times a week".
    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          description: "“Discord monitor” now runs weekdays at 09:00.",
        }),
      ),
    );
  });

  it("says so when there is nothing less often to switch to", async () => {
    server.use(
      getGetScheduledChatCadenceMockHandler({
        ...CADENCE,
        current: "Every Monday at 09:00",
        options: [],
      }),
    );
    await openRunLessOften();

    expect(await screen.findByText(/nothing less often/)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Save" })).toBeNull();
  });

  it("says so when how often it runs can't be read", async () => {
    server.use(
      http.get("*/api/schedules/followups/:scheduleKey/cadence", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );
    await openRunLessOften();

    expect(
      await screen.findByText(/Couldn't load how often it runs/),
    ).toBeDefined();
    expect(screen.queryByRole("radio")).toBeNull();
  });

  it.each([
    [409, "It's paused. Resume it first."],
    [404, "It no longer runs on a schedule this can change."],
  ])(
    "says why when it can't be read to change (%i)",
    async (status, reason) => {
      server.use(
        http.get("*/api/schedules/followups/:scheduleKey/cadence", () =>
          HttpResponse.json({ detail: "refused" }, { status }),
        ),
      );
      await openRunLessOften();

      expect(await screen.findByText(reason)).toBeDefined();
      expect(screen.queryByText(/Couldn't load how often it runs/)).toBeNull();
      expect(screen.queryByRole("button", { name: "Save" })).toBeNull();
    },
  );

  it("asks for a paused chat to be resumed first", async () => {
    server.use(
      getGetScheduledChatCadenceMockHandler(CADENCE),
      http.post("*/api/schedules/followups/:scheduleKey/cadence", () =>
        HttpResponse.json(
          { detail: "This scheduled chat is paused. Resume it first." },
          { status: 409 },
        ),
      ),
    );
    await openRunLessOften();

    fireEvent.click(await screen.findByRole("radio", { name: "Every hour" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
          description: "It's paused. Resume it first.",
        }),
      ),
    );
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
