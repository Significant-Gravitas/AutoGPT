import { getGetBrainDumpSuggestedTasksMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { formatTrialEnd } from "@/components/organisms/TrialCard/helpers";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import {
  afterEach,
  beforeEach,
  expect,
  onTestFinished,
  test,
  vi,
} from "vitest";

const push = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  usePathname: () => "/settings/billing",
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push,
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
}));

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const REASON_TITLE = "Before you go, what's the main reason?";
const TASK: SuggestedTask = {
  id: "task-1",
  title: "Triage this week's open GitHub issues",
  prompt: "Triage our open issues",
  expert_id: "maria",
  state: "open",
};

function setup({ stored = true, tasks = [TASK] } = {}) {
  const cancel = vi.fn(() => trialResponse());
  const patches: unknown[] = [];
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    getPostTrialsCancelTrialMockHandler200(cancel),
    getPostV2ClaimFeedbackPromptMockHandler200({ id: "prompt-1" }),
    getGetBrainDumpSuggestedTasksMockHandler200({ tasks, stored }),
    getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
      patches.push(await request.clone().json());
      return {
        id: "prompt-1",
        kind: "trial_cancel",
        answered: true,
        dismissed: false,
        reason: null,
        score: null,
      };
    }),
  );
  render(<TrialCard />);
  return { cancel, patches };
}

async function openDialog() {
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  return screen.findByRole("dialog", { name: REASON_TITLE });
}

function clickIn(dialog: HTMLElement, name: string) {
  fireEvent.click(within(dialog).getByRole("button", { name }));
}

beforeEach(() => {
  setTrialUser();
  push.mockClear();
  toast.mockClear();
});
afterEach(() => setTrialUser(null));

test("picking a reason saves it", async () => {
  const { patches } = setup();
  clickIn(await openDialog(), "It costs too much");
  await waitFor(() => expect(patches).toEqual([{ reason: "too_expensive" }]));
});

test("Skip reaches the confirmation with no reason saved", async () => {
  const { patches, cancel } = setup();
  clickIn(await openDialog(), "Skip");
  const confirm = await screen.findByRole("dialog", {
    name: "Cancel your trial?",
  });
  expect(patches).toEqual([]);
  clickIn(confirm, "Cancel trial");
  await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
});

test("a failed save still cancels the trial and shows no toast", async () => {
  const { cancel } = setup();
  let finishResponse = () => {};
  const failedResponse = new Promise<void>((resolve) => {
    finishResponse = resolve;
  });
  function observeFailure({
    request,
    response,
  }: {
    request: Request;
    response: Response;
  }) {
    if (request.method === "PATCH" && response.status === 500) finishResponse();
  }
  server.events.on("response:mocked", observeFailure);
  onTestFinished(() =>
    server.events.removeListener("response:mocked", observeFailure),
  );
  server.use(
    http.patch("*/api/feedback-prompts/:id", () =>
      HttpResponse.json({ detail: "boom" }, { status: 500 }),
    ),
  );
  const dialog = await openDialog();
  clickIn(dialog, "It costs too much");
  clickIn(dialog, "Continue");
  clickIn(
    await screen.findByRole("dialog", { name: "Cancel your trial?" }),
    "Cancel trial",
  );
  await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
  await act(async () => {
    await failedResponse;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
  expect(toast).not.toHaveBeenCalled();
});

test("the offer lists open tasks for someone who saw no results", async () => {
  setup({
    tasks: [
      TASK,
      { ...TASK, id: "task-2", title: "Already done", state: "sent" },
    ],
  });
  const dialog = await openDialog();
  clickIn(dialog, "I haven't seen results yet");
  clickIn(dialog, "Continue");
  expect(await screen.findByRole("button", { name: TASK.title })).toBeDefined();
  expect(screen.queryByRole("button", { name: "Already done" })).toBeNull();
});

test("the offer asks for one task when none are stored", async () => {
  setup({ stored: false, tasks: [] });
  const dialog = await openDialog();
  clickIn(dialog, "I haven't seen results yet");
  clickIn(dialog, "Continue");
  expect(
    await screen.findByText(
      "Tell me one weekly task and I'll do it with you now.",
    ),
  ).toBeDefined();
});

test("taking a task starts its chat and keeps the trial", async () => {
  const { cancel, patches } = setup();
  const dialog = await openDialog();
  clickIn(dialog, "I haven't seen results yet");
  clickIn(dialog, "Continue");
  fireEvent.click(await screen.findByRole("button", { name: TASK.title }));
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  expect(push).toHaveBeenCalledWith(
    `/home?autosubmit=true&expertId=maria&suggestedTaskId=task-1#prompt=${encodeURIComponent(TASK.prompt)}`,
  );
  await waitFor(() => expect(patches).toContainEqual({ offer_taken: true }));
  expect(cancel).not.toHaveBeenCalled();
});

test("the confirmation shows the real end date", async () => {
  setup();
  clickIn(await openDialog(), "Skip");
  const confirm = await screen.findByRole("dialog", {
    name: "Cancel your trial?",
  });
  expect(
    within(confirm).getByText(formatTrialEnd(trialResponse().ends_at)).tagName,
  ).toBe("STRONG");
});

test("reopening cancellation claims a new showing without the old note", async () => {
  setup();
  let claims = 0;
  const saves: { id: string; body: unknown }[] = [];
  server.use(
    getPostV2ClaimFeedbackPromptMockHandler200(() => ({
      id: `prompt-${++claims}`,
    })),
    getPatchV2UpdateFeedbackPromptMockHandler200(
      async ({ request, params }) => {
        saves.push({
          id: String(params.id),
          body: await request.clone().json(),
        });
        return {
          id: String(params.id),
          kind: "trial_cancel",
          answered: true,
          dismissed: false,
          reason: null,
          score: null,
        };
      },
    ),
  );
  let dialog = await openDialog();
  clickIn(dialog, "Something else");
  fireEvent.change(screen.getByRole("textbox"), {
    target: { value: "First attempt" },
  });
  clickIn(dialog, "Continue");
  clickIn(
    await screen.findByRole("dialog", { name: "Cancel your trial?" }),
    "Keep trial",
  );
  await waitFor(() =>
    expect(saves).toContainEqual({
      id: "prompt-1",
      body: { note: "First attempt" },
    }),
  );
  dialog = await openDialog();
  clickIn(dialog, "It costs too much");
  await waitFor(() =>
    expect(saves).toContainEqual({
      id: "prompt-2",
      body: { reason: "too_expensive" },
    }),
  );
  expect(claims).toBe(2);
});
