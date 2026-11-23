import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListExpertsMockHandler200,
  getListExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getListTaskSuggestionsMockHandler200,
  getListTasksMockHandler200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { BriefingTasks } from "@/app/api/__generated__/models/briefingTasks";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { useWorkAreaStore } from "../../../workArea";
import { BriefTab } from "../BriefTab";
import { UNDO_MS } from "../SuggestedNext/usePendingDismiss";

const [anaMock] = getListExpertsResponseMock200();
const ANA = { ...anaMock, id: "expert-ana", name: "Ana" };
const API = "/api/proxy/api/tasks";

function suggestion(id: string, title: string, reason: string): Task {
  return {
    ...makeTask(id, title, "TODO", ANA.id),
    created_by: "OTTO",
    is_suggested: true,
    suggested_reason: reason,
  };
}

const DECK = suggestion(
  "s-deck",
  "Turn the pricing comparison into a deck",
  "The pricing comparison finished yesterday; this turns it into the deck you mentioned.",
);
const CALLS = suggestion(
  "s-calls",
  "Book calls with the two cheapest vendors",
  "The shortlist has waited four days.",
);

let suggestions: Task[];
let calls: string[];

function serve({
  tasks = [] as Task[],
  storedTasks = null as BriefingTasks | null,
} = {}) {
  const dashboard = makeDashboard([]);
  server.use(
    getListExpertsMockHandler200([ANA]),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getListTasksMockHandler200({ tasks }),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json({
        ...dashboard,
        briefing: { ...dashboard.briefing, tasks: storedTasks },
      }),
    ),
    getListTaskSuggestionsMockHandler200(() => ({ tasks: suggestions })),
    http.post(`${API}/:taskID/run`, ({ params }) => {
      calls.push(`run ${params.taskID}`);
      const task = suggestions.find((entry) => entry.id === params.taskID)!;
      suggestions = suggestions.filter((entry) => entry.id !== task.id);
      return HttpResponse.json({
        ...task,
        is_suggested: false,
        status: "IN_PROGRESS",
      });
    }),
    http.post(`${API}/:taskID/dismiss`, ({ params }) => {
      calls.push(`dismiss ${params.taskID}`);
      const task = suggestions.find((entry) => entry.id === params.taskID)!;
      suggestions = suggestions.filter((entry) => entry.id !== task.id);
      return HttpResponse.json(task);
    }),
    http.post(`${API}/suggestions/refresh`, () => {
      calls.push("refresh");
      suggestions = [...suggestions, CALLS];
      return HttpResponse.json({ tasks: suggestions });
    }),
  );
}

beforeEach(() => {
  suggestions = [DECK];
  calls = [];
  useWorkAreaStore.getState().reset();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("Brief — Suggested next", () => {
  it("shows each drafted task with its reason and owner", async () => {
    serve();
    render(<BriefTab />);

    expect(
      await screen.findByText("Turn the pricing comparison into a deck"),
    ).toBeDefined();
    expect(
      screen.getByRole("region", { name: "Suggested next" }),
    ).toBeDefined();
    expect(screen.getByText(DECK.suggested_reason!)).toBeDefined();
    expect(screen.getByRole("img", { name: "Ana" })).toBeDefined();
  });

  it("Start runs the task, and the card then opens the running task", async () => {
    const user = userEvent.setup();
    serve();
    render(<BriefTab />);

    await user.click(
      await screen.findByRole("button", { name: `Start ${DECK.title}` }),
    );

    expect(await screen.findByText("Ana is on it.")).toBeDefined();
    expect(calls).toEqual([`run ${DECK.id}`]);
    await user.click(screen.getByRole("button", { name: "Open" }));
    expect(useWorkAreaStore.getState().activeKey).toBe(`task:${DECK.id}`);
  });

  it("Not now takes the card away with an undo, and sends it once the undo runs out", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    serve();
    render(<BriefTab />);

    await user.click(
      await screen.findByRole("button", { name: `Not now: ${DECK.title}` }),
    );
    expect(screen.queryByText(DECK.title)).toBeNull();
    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(screen.getByText(DECK.title)).toBeDefined();
    expect(calls).toEqual([]);

    await user.click(
      screen.getByRole("button", { name: `Not now: ${DECK.title}` }),
    );
    await act(() => vi.advanceTimersByTimeAsync(UNDO_MS));

    await waitFor(() => expect(calls).toEqual([`dismiss ${DECK.id}`]));
    expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
  });

  it("Suggest more drafts more and shows them", async () => {
    const user = userEvent.setup();
    serve();
    render(<BriefTab />);

    await user.click(
      await screen.findByRole("button", { name: "Suggest more" }),
    );

    expect(await screen.findByText(CALLS.title)).toBeDefined();
    expect(calls).toEqual(["refresh"]);
  });

  it("leaves the block out when nothing is drafted", async () => {
    suggestions = [];
    serve();
    render(<BriefTab />);

    expect(
      await screen.findByRole("button", { name: "Suggest tasks" }),
    ).toBeDefined();
    expect(screen.queryByRole("region", { name: "Suggested next" })).toBeNull();
  });
});

describe("Brief — tasks", () => {
  const waiting = makeTask(
    "t-venue",
    "Pick a venue for the offsite",
    "NEEDS_INPUT",
  );

  it("reads the tasks this morning's brief wrote down", async () => {
    serve({
      tasks: [waiting],
      storedTasks: {
        finished: [
          {
            task_id: "t-pricing",
            title: "Compare vendor pricing",
            status: "DONE",
            owner_name: "Ana",
            line: "Plan B is 20% cheaper.",
          },
        ],
        failed: [],
        waiting: [],
      },
    });
    render(<BriefTab />);

    expect(
      await screen.findByRole("region", {
        name: "Finished since your last brief",
      }),
    ).toBeDefined();
    expect(screen.getByText("Plan B is 20% cheaper.")).toBeDefined();
    expect(screen.queryByRole("region", { name: "Needs you" })).toBeNull();

    await userEvent.setup().click(screen.getByText("Compare vendor pricing"));
    expect(useWorkAreaStore.getState().activeKey).toBe("task:t-pricing");
  });

  it("joins the tasks in itself for a brief written before it covered them", async () => {
    serve({ tasks: [waiting] });
    render(<BriefTab />);

    expect(
      await screen.findByRole("region", { name: "Needs you" }),
    ).toBeDefined();
    expect(
      screen.getAllByText("Pick a venue for the offsite").length,
    ).toBeGreaterThan(0);
    expect(
      screen.queryByRole("region", { name: "Finished since your last brief" }),
    ).toBeNull();
  });
});
