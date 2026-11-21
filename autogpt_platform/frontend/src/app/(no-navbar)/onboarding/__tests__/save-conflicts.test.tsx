import {
  getGetV1OnboardingStateResponseMock,
  getPatchV1UpdateOnboardingStateResponseMock,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import type { UserOnboardingUpdate } from "@/app/api/__generated__/models/userOnboardingUpdate";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import * as Sentry from "@sentry/nextjs";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import OnboardingPage from "../page";
import { readLocalProgress } from "../progress";
import { useOnboardingWizardStore } from "../store";
import { landDraftSave, type StoredDraft } from "./draft-save-rule";
import { makeProgress } from "./progress-fixture";

// #17158. A save the wizard stops waiting for at 10 s can still reach the
// server and land later. The wizard's next save carried the revision from
// before it, and the server refused it as if another session had saved
// (Sentry BUILDER-962).

vi.mock("@sentry/nextjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@sentry/nextjs")>()),
  captureMessage: vi.fn(),
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
const { router, params, user } = vi.hoisted(() => ({
  router: {
    replace: () => {},
    push: () => {},
    refresh: () => {},
    prefetch: () => {},
    back: () => {},
    forward: () => {},
  },
  params: new URLSearchParams(),
  user: { id: "u1", email: "pat@example.com", user_metadata: {} } as User,
}));
vi.mock("next/navigation", () => ({
  useRouter: () => router,
  useSearchParams: () => params,
  usePathname: () => "/onboarding",
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: true,
    isUserLoading: false,
    user,
    refreshSession: () => {},
  }),
}));

const REFUSED = "Request failed on client: PATCH /api/onboarding 409";
type Delivery = "answered" | "landsUnanswered" | "landsLate" | "landsTwice";

let stored: StoredDraft<OnboardingWizardProgress>;
let deliveries: Delivery[];
let held: Array<() => void>;
let landed: number;

function land(save: UserOnboardingUpdate) {
  landed += 1;
  const next = landDraftSave(stored, save);
  if (!next)
    return HttpResponse.json(
      {
        detail:
          "Onboarding progress changed in another session. Reload before saving.",
      },
      { status: 409 },
    );
  stored = next;
  return HttpResponse.json(
    getPatchV1UpdateOnboardingStateResponseMock({
      userId: user.id,
      completedSteps: [],
      wizardProgress: stored.progress,
      wizardRevision: stored.revision,
    }),
  );
}

function hold() {
  return new Promise<void>((release) => held.push(release));
}

beforeEach(() => {
  stored = { progress: null, revision: 0, writerId: null, writeSeq: null };
  deliveries = [];
  held = [];
  landed = 0;
  localStorage.clear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  // Read by the flags behind the wizard's layout.
  useAuthStore.setState({ user, hasLoadedUser: true, isUserLoading: false });
  resetClientRequestFailureLog();
  vi.mocked(Sentry.captureMessage).mockClear();
  server.use(
    http.get("*/api/onboarding", () =>
      HttpResponse.json(
        getGetV1OnboardingStateResponseMock({
          userId: user.id,
          completedSteps: [],
          wizardProgress: stored.progress,
          wizardRevision: stored.revision,
        }),
      ),
    ),
    http.patch("*/api/onboarding", async ({ request }) => {
      const save = (await request.json()) as UserOnboardingUpdate;
      const delivery = deliveries.shift() ?? "answered";
      if (delivery === "landsLate") await hold();
      if (delivery === "landsTwice") land(save);
      const answer = land(save);
      if (delivery === "landsUnanswered") await hold();
      return answer;
    }),
  );
});
afterEach(() => {
  held.forEach((release) => release());
  vi.useRealTimers();
  useAuthStore.setState({
    user: null,
    hasLoadedUser: false,
    isUserLoading: true,
  });
});

async function openWizard() {
  render(<OnboardingPage />);
  await screen.findByRole("button", { name: "Meet Otto" });
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

function refusalsReported() {
  return vi
    .mocked(Sentry.captureMessage)
    .mock.calls.filter(([message]) => message === REFUSED);
}

function savedLocally(currentStep: string, revision: number) {
  expect(readLocalProgress(user.id)).toEqual({
    progress: expect.objectContaining({ currentStep }),
    revision,
    pending: false,
  });
}

it("saves the next answer over a save it stopped waiting for, without a refusal", async () => {
  deliveries = ["landsUnanswered"];
  await openWizard();

  fireEvent.click(screen.getByRole("button", { name: "Meet Otto" }));
  await advance(200);
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  await advance(10_000);
  expect(screen.getByText(/couldn't save your progress/i)).toBeDefined();
  await advance(2_000);
  vi.useRealTimers();

  await waitFor(() => savedLocally("role", 2));
  expect(refusalsReported()).toEqual([]);
  expect(stored).toMatchObject({
    progress: { currentStep: "role" },
    revision: 2,
  });
  expect(screen.queryByText(/updated in another session/i)).toBeNull();
  expect(screen.queryByText(/couldn't save your progress/i)).toBeNull();
});

it("doesn't refuse a save that reaches the server twice", async () => {
  deliveries = ["landsTwice"];
  render(<OnboardingPage />);

  fireEvent.click(await screen.findByRole("button", { name: "Meet Otto" }));

  await waitFor(() => expect(readLocalProgress(user.id)?.pending).toBe(false));
  expect(refusalsReported()).toEqual([]);
  savedLocally("autopilot", 2);
  expect(screen.queryByText(/updated in another session/i)).toBeNull();
});

it("never lets a save it stopped waiting for replace a newer one when it lands late", async () => {
  deliveries = ["landsLate"];
  await openWizard();

  fireEvent.click(screen.getByRole("button", { name: "Meet Otto" }));
  await advance(200);
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  await advance(12_000);
  vi.useRealTimers();
  await waitFor(() => savedLocally("role", 1));

  await act(async () => held.forEach((release) => release()));

  await waitFor(() => expect(landed).toBe(2));
  expect(stored).toMatchObject({
    progress: { currentStep: "role" },
    revision: 1,
  });
  expect(refusalsReported()).toEqual([]);
});

it("still shows the conflict when another tab saved in between", async () => {
  render(<OnboardingPage />);
  await screen.findByRole("button", { name: "Meet Otto" });
  stored = {
    progress: makeProgress({ role: "Saved in another tab" }),
    revision: 1,
    writerId: "another-tab",
    writeSeq: 1,
  };

  fireEvent.click(screen.getByRole("button", { name: "Meet Otto" }));

  expect(await screen.findByText(/updated in another session/i)).toBeDefined();
  expect(refusalsReported()).toHaveLength(1);
  expect(stored.progress?.role).toBe("Saved in another tab");
});
