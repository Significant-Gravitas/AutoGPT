import { getListExpertFirstJobsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { ExpertFirstJob } from "@/app/api/__generated__/models/expertFirstJob";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TeamAtWork } from "../TeamAtWork";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

const AVA: ExpertFirstJob = {
  expert_id: "expert-ava",
  name: "Ava",
  role: "Inbox manager",
  avatar_url: null,
  title: "Sort this week's inbox",
  description: "Label every unread email and flag what needs you.",
  status: "open",
  detail: "Not started yet",
  providers: [],
  session_id: null,
};

const BEN: ExpertFirstJob = {
  expert_id: "expert-ben",
  name: "Ben",
  role: "Researcher",
  avatar_url: null,
  title: "Map the top five competitors",
  description: "Pricing, positioning and recent launches for each.",
  status: "working",
  detail: "Working on it now",
  providers: [],
  session_id: "session-ben",
};

const CLEO: ExpertFirstJob = {
  expert_id: "expert-cleo",
  name: "Cleo",
  role: "Scheduler",
  avatar_url: null,
  title: "Tidy next week's calendar",
  description: "Move clashing meetings and protect focus time.",
  status: "needs_setup",
  detail: "Needs google, notion",
  providers: ["google", "notion"],
  session_id: null,
};

function serve(items: ExpertFirstJob[]) {
  server.use(getListExpertFirstJobsMockHandler200({ items }));
}

function events(name: string) {
  return capture.mock.calls
    .filter(([event]) => event === name)
    .map(([, properties]) => properties);
}

beforeEach(() => {
  capture.mockReset();
});

describe("TeamAtWork", () => {
  it("renders one card per first job with its status label", async () => {
    serve([AVA, BEN, CLEO]);
    render(<TeamAtWork />);

    expect(
      await screen.findByRole("heading", { name: "Your team" }),
    ).toBeDefined();
    expect(
      screen.getByRole("link", {
        name: "Ava: Sort this week's inbox, Start a chat",
      }),
    ).toBeDefined();
    const ben = screen.getByRole("link", {
      name: "Ben: Map the top five competitors, Working",
    });
    expect(ben.getAttribute("href")).toBe("/copilot?sessionId=session-ben");
    const cleo = screen.getByRole("link", {
      name: "Cleo: Tidy next week's calendar, Needs setup",
    });
    expect(cleo.getAttribute("href")).toBe("/team/expert-cleo");
    expect(screen.getByText("Needs Google, Notion")).toBeDefined();
    expect(screen.queryByText("Needs google, notion")).toBeNull();
    expect(screen.getByText("Researcher")).toBeDefined();
    expect(
      screen.getByText("Pricing, positioning and recent launches for each."),
    ).toBeDefined();
    expect(screen.getAllByRole("link")).toHaveLength(3);
  });

  it("links an unstarted job to the expert kickoff", async () => {
    serve([AVA]);
    render(<TeamAtWork />);

    const card = await screen.findByRole("link", {
      name: "Ava: Sort this week's inbox, Start a chat",
    });
    expect(card.getAttribute("href")).toBe(
      "/copilot?expertId=expert-ava&kickoff=1",
    );

    await userEvent.click(card);

    expect(events("team_first_job_opened")).toEqual([
      { expert_id: "expert-ava", status: "open" },
    ]);
  });

  it("renders nothing when there are no experts", async () => {
    let reads = 0;
    server.use(
      http.get("*/api/experts/first-jobs", () => {
        reads += 1;
        return HttpResponse.json({ items: [] });
      }),
    );
    render(<TeamAtWork />);

    await waitFor(() => expect(reads).toBe(1));
    await act(async () => {});
    expect(screen.queryByRole("heading", { name: "Your team" })).toBeNull();
    expect(screen.queryByTestId("team-at-work")).toBeNull();
  });

  describe("while an expert is working", () => {
    afterEach(() => {
      vi.useRealTimers();
    });

    async function advance(ms: number) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(ms);
      });
    }

    it("polls until the job is ready, then stops", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      let reads = 0;
      let current: ExpertFirstJob = BEN;
      server.use(
        http.get("*/api/experts/first-jobs", () => {
          reads += 1;
          return HttpResponse.json({ items: [current] });
        }),
      );
      render(<TeamAtWork />);

      await screen.findByRole("link", {
        name: "Ben: Map the top five competitors, Working",
      });
      const readsBefore = reads;
      current = { ...BEN, status: "ready", detail: "Ready for review" };

      await advance(5_000);

      expect(
        await screen.findByRole("link", {
          name: "Ben: Map the top five competitors, Ready for review",
        }),
      ).toBeDefined();
      expect(reads).toBe(readsBefore + 1);

      await advance(10_000);
      expect(reads).toBe(readsBefore + 1);
    });
  });
});
