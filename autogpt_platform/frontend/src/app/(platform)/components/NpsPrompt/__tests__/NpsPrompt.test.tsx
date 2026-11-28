import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { beforeEach, expect, test, vi } from "vitest";
import { NpsPrompt } from "../NpsPrompt";
import { isNpsDue } from "../helpers";

const auth = vi.hoisted(() => ({
  user: null as { id: string; created_at: string } | null,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user, isLoggedIn: Boolean(auth.user) }),
}));

const DAY_MS = 24 * 60 * 60 * 1000;
const QUESTION =
  "How likely are you to recommend AutoGPT to a friend or colleague?";

function daysAgo(days: number) {
  return new Date(Date.now() - days * DAY_MS).toISOString();
}

function tree() {
  return <NpsPrompt />;
}

function setup(claimedId: string | null = "prompt-1") {
  const claims: unknown[] = [];
  const patches: unknown[] = [];
  server.use(
    getPostV2ClaimFeedbackPromptMockHandler200(async ({ request }) => {
      claims.push(await request.clone().json());
      return { id: claimedId };
    }),
    getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
      patches.push(await request.clone().json());
      return {
        id: "prompt-1",
        kind: "nps",
        answered: true,
        dismissed: false,
        reason: null,
        score: null,
      };
    }),
  );
  return { claims, patches };
}

beforeEach(() => {
  window.localStorage.clear();
  auth.user = { id: "user-a", created_at: daysAgo(15) };
});

test("an account under 14 days old makes no claim", async () => {
  auth.user = { id: "user-a", created_at: daysAgo(13) };
  const { claims } = setup();
  render(tree());
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(claims).toEqual([]);
  expect(screen.queryByText(QUESTION)).toBeNull();
});

test("an account at 14 days or more claims and sees the question", async () => {
  const { claims } = setup();
  render(tree());
  expect(await screen.findByText(QUESTION)).toBeDefined();
  expect(claims).toEqual([{ kind: "nps" }]);
});

test("a refused claim waits a day before asking the backend again", async () => {
  const { claims } = setup(null);
  const { unmount } = render(tree());
  await waitFor(() =>
    expect(storage.get(Key.NPS_CHECK)).toMatch(/^user-a\|\d+$/),
  );
  expect(screen.queryByText(QUESTION)).toBeNull();
  unmount();
  render(tree());
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(claims).toHaveLength(1);
});

test("a refusal from a day ago is checked again", async () => {
  storage.set(Key.NPS_CHECK, `user-a|${Date.now() - 1}`);
  const { claims } = setup();
  render(tree());
  expect(await screen.findByText(QUESTION)).toBeDefined();
  expect(claims).toEqual([{ kind: "nps" }]);
  expect(storage.get(Key.NPS_CHECK)).toBe("user-a");
});

test("a failed claim leaves no marker, so the next load asks again", async () => {
  let attempts = 0;
  server.use(
    http.post("*/api/internal/feedback-prompts/claim", () => {
      attempts += 1;
      return HttpResponse.json({ detail: "boom" }, { status: 500 });
    }),
  );
  const { unmount } = render(tree());
  await waitFor(() => expect(attempts).toBe(1));
  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(storage.get(Key.NPS_CHECK)).toBeFalsy();
  expect(screen.queryByText(QUESTION)).toBeNull();
  unmount();
  render(tree());
  await waitFor(() => expect(attempts).toBe(2));
});

test("once asked on this browser no claim happens", async () => {
  storage.set(Key.NPS_CHECK, "user-a");
  const { claims } = setup();
  render(tree());
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(claims).toEqual([]);
});

test("choosing a score saves it", async () => {
  const { patches } = setup();
  render(tree());
  await screen.findByText(QUESTION);
  fireEvent.click(screen.getByRole("button", { name: "9" }));
  await waitFor(() => expect(patches).toEqual([{ score: 9 }]));
});

test("closing saves a dismissal", async () => {
  const { patches } = setup();
  render(tree());
  await screen.findByText(QUESTION);
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  await waitFor(() => expect(patches).toEqual([{ dismissed: true }]));
});

test("the score buttons and labels use the agreed copy", async () => {
  setup();
  render(tree());
  await screen.findByText(QUESTION);
  expect(screen.getByText("Not at all likely")).toBeDefined();
  expect(screen.getByText("Extremely likely")).toBeDefined();
  fireEvent.click(screen.getByRole("button", { name: "3" }));
  expect(
    await screen.findByText("What's the main thing we should fix?"),
  ).toBeDefined();
});

test("NPS becomes due at exactly fourteen days", () => {
  const created = "2026-11-01T12:00:00Z";
  const due = Date.parse(created) + 14 * DAY_MS;
  expect(isNpsDue(created, due - 1)).toBe(false);
  expect(isNpsDue(created, due)).toBe(true);
});
