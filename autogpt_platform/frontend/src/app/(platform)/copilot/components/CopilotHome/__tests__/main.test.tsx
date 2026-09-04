import { expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getGetBriefingsGetLatestBriefingMockHandler200 } from "@/app/api/__generated__/endpoints/briefings/briefings.msw";
import { EmptySession } from "../../EmptySession/EmptySession";

// Rendered through EmptySession, not CopilotHome directly: the briefing recap
// mounts inside it, and the onboarding surface + composer recipient picker
// that live there must survive the experts flag being on.
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.HIRE_EXPERTS || flag === actual.Flag.AGENT_BRIEFING,
  };
});

const baseProps = {
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onSend: vi.fn(),
};

// The recap renders nothing until its briefing request settles, so tests
// that assert its absence wait for the mocked response rather than trusting
// a blank screen that may only mean "still loading".
function trackResponses() {
  const paths: string[] = [];
  function record({ request }: { request: Request }) {
    paths.push(new URL(request.url).pathname);
  }
  server.events.on("response:mocked", record);
  return {
    paths,
    briefingSettled: () =>
      waitFor(() =>
        expect(paths.some((path) => path.includes("briefings"))).toBe(true),
      ),
    stop: () => server.events.removeListener("response:mocked", record),
  };
}

test("renders greeting and composer", async () => {
  render(<EmptySession {...baseProps} />);
  expect(await screen.findByPlaceholderText(/./)).toBeDefined();
});

test("shows a named kickoff status and withholds the empty composer", () => {
  render(<EmptySession {...baseProps} isKickoffStarting expertName="Maria" />);

  expect(screen.getByRole("status").textContent).toContain(
    "Opening Maria's workspace",
  );
  expect(screen.queryByPlaceholderText(/./)).toBeNull();
});

test("leaves the space under the composer empty when there is no briefing", async () => {
  // The workflow-runs strip that used to fill this gap lives on /home now,
  // under the briefing tile.
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  const responses = trackResponses();

  try {
    render(<EmptySession {...baseProps} />);
    expect(await screen.findByPlaceholderText(/./)).toBeDefined();
    await responses.briefingSettled();

    expect(screen.queryByText("Recap")).toBeNull();
    expect(screen.queryByText("What's happening with your agents")).toBeNull();
  } finally {
    responses.stop();
  }
});

test("renders briefing sections when a briefing is available", async () => {
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: new Date(),
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: [
          {
            expert_id: "expert-1",
            expert_name: "Sales Scout",
            expert_avatar_url: null,
            agent_name: "Lead Finder",
            graph_id: "graph-1",
            execution_id: "exec-1",
            library_agent_id: "lib-1",
            status: "COMPLETED",
            summary: "Found 3 leads",
            link: "/library/agents/lib-1/runs/exec-1",
          },
          {
            expert_id: "expert-2",
            expert_name: "Support Bot",
            expert_avatar_url: null,
            agent_name: "Ticket Triager",
            graph_id: "graph-2",
            execution_id: "exec-2",
            library_agent_id: "lib-2",
            status: "FAILED",
            summary: null,
            link: "/library/agents/lib-2/runs/exec-2",
          },
        ],
        decision_items: [
          {
            node_exec_id: "node-1",
            graph_exec_id: "graph-exec-1",
            title: "Approve outreach email",
            expert_id: "expert-1",
            expert_name: "Sales Scout",
            expert_avatar_url: null,
            link: "/library/agents/lib-1/runs/exec-1?node=node-1",
          },
        ],
      },
    }),
  );
  render(<EmptySession {...baseProps} />);

  expect(await screen.findByText("Recap")).toBeDefined();
  expect(screen.getByText("Lead Finder")).toBeDefined();
  expect(screen.getByText("Ticket Triager")).toBeDefined();
  // A summary carries the expert it came from, matching the thread markdown.
  const finding = screen.getByText(/Found 3 leads/);
  expect(finding.textContent).toContain("Sales Scout");

  // The recap does not repeat the decisions: the needs-you list on /home
  // shows the same pending reviews, and it can act on them.
  expect(screen.queryByText(/Needs your decision/)).toBeNull();
});

test("shows three runs, then all of them behind the show-all toggle", async () => {
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: null,
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: Array.from({ length: 5 }, (_, i) => ({
          expert_id: null,
          expert_name: null,
          expert_avatar_url: null,
          agent_name: `Agent ${i}`,
          graph_id: `graph-${i}`,
          execution_id: `exec-${i}`,
          library_agent_id: null,
          status: "COMPLETED",
          summary: null,
          link: null,
        })),
        decision_items: [],
        decision_total: 0,
      },
    }),
  );
  render(<EmptySession {...baseProps} />);

  // Every row stays mounted so the card can animate its height in both
  // directions; the collapsed card clips to three rows visually, and only
  // the expanded one scrolls.
  expect(await screen.findByText("Agent 4")).toBeDefined();
  expect(screen.getByRole("list").className).toContain("overflow-hidden");

  await userEvent.click(
    screen.getByRole("button", { name: /Show all results \(5\)/ }),
  );

  expect(screen.getByRole("list").className).toContain("overflow-y-auto");
  expect(screen.getByRole("button", { name: /Show less/ })).toBeDefined();
});

test("links each row straight at its run", async () => {
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: null,
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: [
          {
            expert_id: "expert-1",
            expert_name: "Sales Scout",
            expert_avatar_url: null,
            agent_name: "Lead Finder",
            graph_id: "graph-1",
            execution_id: "exec-1",
            library_agent_id: "lib-1",
            status: "COMPLETED",
            summary: "Found 3 leads",
            link: "/library/agents/lib-1/runs/exec-1",
          },
        ],
        decision_items: [],
        decision_total: 0,
      },
    }),
  );
  render(<EmptySession {...baseProps} />);

  const row = await screen.findByRole("link", { name: /Lead Finder/ });
  expect(row.getAttribute("href")).toBe("/library/agents/lib-1/runs/exec-1");
});

test("keeps the decisions inbox and team status off the copilot recap", async () => {
  // Both moved to /home. Proving the pending-reviews query never fires is
  // what keeps the inbox from creeping back in: absent text alone would
  // also pass while the request was still in flight. (The experts query has
  // no such tell — the composer's recipient picker still needs it.)
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  const responses = trackResponses();

  try {
    render(<EmptySession {...baseProps} />);
    await screen.findByPlaceholderText(/./);
    await responses.briefingSettled();

    expect(responses.paths.some((path) => path.includes("review"))).toBe(false);
    expect(screen.queryByText(/Needs your attention/)).toBeNull();
    expect(screen.queryByRole("link", { name: /Chat with/ })).toBeNull();
  } finally {
    responses.stop();
  }
});

test("shows an error card when the briefing fetch fails", async () => {
  server.use(
    http.get("/api/proxy/api/briefings/latest", () =>
      HttpResponse.json({ detail: "boom" }, { status: 500 }),
    ),
  );
  render(<EmptySession {...baseProps} />);

  expect(await screen.findByText("Failed to load your briefing")).toBeDefined();
});

test("keeps the onboarding surface and recipient picker with the experts flag on", async () => {
  // hire-experts and onboarding-brain-dump are independent flags aimed at
  // overlapping beta cohorts; the briefing recap must not cancel the other
  // rollout out from under a brand-new user.
  server.use(getGetBriefingsGetLatestBriefingMockHandler200(null));
  render(<EmptySession {...baseProps} />);

  // The composer's expert picker is the only way to address an expert from
  // the home, and it only renders behind the experts flag.
  expect(await screen.findByText("Autopilot")).toBeDefined();
  // Suggestion themes come from EmptySession, not the briefing block.
  expect(screen.getByPlaceholderText(/./)).toBeDefined();
});

test("renders a run row without a link when the briefing has no deep link", async () => {
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: null,
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: [
          {
            expert_id: null,
            expert_name: null,
            expert_avatar_url: null,
            agent_name: "Unlinked Agent",
            graph_id: "graph-9",
            execution_id: "exec-9",
            library_agent_id: null,
            status: "COMPLETED",
            summary: null,
            link: null,
          },
        ],
        decision_items: [],
        decision_total: 0,
      },
    }),
  );
  render(<EmptySession {...baseProps} />);

  const row = await screen.findByText("Unlinked Agent");
  expect(row.closest("a")).toBeNull();
});

test("does not render a hollow briefing card when there are no runs", async () => {
  // A run paused on an approval is not terminal, so it never lands in the
  // run list — leaving a card that would show only a date.
  const responses = trackResponses();
  server.use(
    getGetBriefingsGetLatestBriefingMockHandler200({
      id: "briefing-1",
      briefing_date: new Date(),
      created_at: new Date(),
      delivered_at: null,
      content: {
        generated_at: new Date(),
        timezone: "UTC",
        zero_expert_fallback: false,
        run_items: [],
        decision_items: [
          {
            node_exec_id: "node-1",
            graph_exec_id: "graph-exec-1",
            title: "Approve outreach email",
            expert_id: null,
            expert_name: null,
            expert_avatar_url: null,
            link: "/library",
          },
        ],
        decision_total: 1,
      },
    }),
  );

  try {
    render(<EmptySession {...baseProps} />);
    await screen.findByPlaceholderText(/./);
    await responses.briefingSettled();

    // The decisions inbox that used to carry this case is on /home now, so
    // the empty state stays blank rather than showing a dated, empty card.
    expect(screen.queryByText("This morning")).toBeNull();
    expect(screen.queryByText("Recap")).toBeNull();
  } finally {
    responses.stop();
  }
});
