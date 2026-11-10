import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
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

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  usePathname: () => "/settings/billing",
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({
    push: vi.fn(),
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

const TITLE = "Cancel your trial?";

function setup() {
  const cancels: unknown[] = [];
  const patches: unknown[] = [];
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    http.post("*/api/internal/credits/trial/cancel", async ({ request }) => {
      cancels.push(await request.clone().json());
      return HttpResponse.json(trialResponse());
    }),
    getPostV2ClaimFeedbackPromptMockHandler200({ id: "prompt-1" }),
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
  return { cancels, patches };
}

async function openDialog() {
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  return screen.findByRole("dialog", { name: TITLE });
}

function clickIn(dialog: HTMLElement, name: string) {
  fireEvent.click(within(dialog).getByRole("button", { name }));
}

beforeEach(() => {
  setTrialUser();
  toast.mockClear();
});
afterEach(() => setTrialUser(null));

test("one dialog confirms, shows the real end date and keeps both buttons", async () => {
  setup();
  const dialog = await openDialog();
  expect(
    within(dialog).getByText(formatTrialEnd(trialResponse().ends_at)).tagName,
  ).toBe("STRONG");
  clickIn(dialog, "It costs too much");
  expect(
    within(dialog).getByRole("button", { name: "Keep trial" }),
  ).toBeDefined();
  expect(
    within(dialog).getByRole("button", { name: "Cancel trial" }),
  ).toBeDefined();
  expect(within(dialog).queryByRole("button", { name: "Skip" })).toBeNull();
  expect(within(dialog).queryByRole("button", { name: "Continue" })).toBeNull();
});

test("cancelling without a reason sends none", async () => {
  const { cancels, patches } = setup();
  clickIn(await openDialog(), "Cancel trial");
  await waitFor(() =>
    expect(cancels).toEqual([{ reason: null, comment: null }]),
  );
  expect(patches).toEqual([]);
});

test("the picked reason and note reach our feedback and the cancellation", async () => {
  const { cancels, patches } = setup();
  const dialog = await openDialog();
  clickIn(dialog, "It's missing something");
  fireEvent.change(
    within(dialog).getByRole("textbox", {
      name: "Tell us a bit more (optional)",
    }),
    { target: { value: "  A Slack integration " } },
  );
  clickIn(dialog, "Cancel trial");
  await waitFor(() =>
    expect(cancels).toEqual([
      { reason: "missing_something", comment: "A Slack integration" },
    ]),
  );
  await waitFor(() =>
    expect(patches).toEqual([
      { reason: "missing_something" },
      { note: "A Slack integration" },
    ]),
  );
});

test("keeping the trial cancels nothing", async () => {
  const { cancels, patches } = setup();
  const dialog = await openDialog();
  clickIn(dialog, "I haven't seen results yet");
  clickIn(dialog, "Keep trial");
  await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  await waitFor(() => expect(patches).toEqual([{ reason: "no_results_yet" }]));
  expect(cancels).toEqual([]);
});

test("a failed save still cancels the trial and shows no toast", async () => {
  const { cancels } = setup();
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
    http.patch("*/api/internal/feedback-prompts/:id", () =>
      HttpResponse.json({ detail: "boom" }, { status: 500 }),
    ),
  );
  const dialog = await openDialog();
  clickIn(dialog, "It costs too much");
  clickIn(dialog, "Cancel trial");
  await waitFor(() =>
    expect(cancels).toEqual([{ reason: "too_expensive", comment: null }]),
  );
  await act(async () => {
    await failedResponse;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
  expect(toast).not.toHaveBeenCalled();
});

test("reopening cancellation claims a new showing without the old answer", async () => {
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
  clickIn(dialog, "Keep trial");
  await waitFor(() =>
    expect(saves).toContainEqual({ id: "prompt-1", body: { reason: "other" } }),
  );
  dialog = await openDialog();
  expect(within(dialog).queryByRole("textbox")).toBeNull();
  clickIn(dialog, "It costs too much");
  await waitFor(() =>
    expect(saves).toContainEqual({
      id: "prompt-2",
      body: { reason: "too_expensive" },
    }),
  );
  expect(claims).toBe(2);
});
