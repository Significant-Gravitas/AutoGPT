import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { TopUpPromptContext } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, expect, test, vi } from "vitest";
import { NpsPrompt } from "../NpsPrompt";

const auth = vi.hoisted(() => ({
  user: null as { id: string; created_at: string } | null,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: auth.user, isLoggedIn: Boolean(auth.user) }),
}));

const DAY_MS = 24 * 60 * 60 * 1000;
const QUESTION = "How likely are you to recommend AutoGPT?";

function daysAgo(days: number) {
  return new Date(Date.now() - days * DAY_MS).toISOString();
}

function tree(isTopUpCardOpen = false) {
  return (
    <TopUpPromptContext.Provider
      value={{
        isOutOfCredits: false,
        openTopUp: () => {},
        closeTopUp: () => {},
        isTopUpCardOpen,
      }}
    >
      <NpsPrompt />
    </TopUpPromptContext.Provider>
  );
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

test("a refused claim leaves the marker and shows nothing", async () => {
  setup(null);
  render(tree());
  await waitFor(() => expect(storage.get(Key.NPS_DONE)).toBe("user-a"));
  expect(screen.queryByText(QUESTION)).toBeNull();
});

test("with the marker set no claim happens", async () => {
  storage.set(Key.NPS_DONE, "user-a");
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

test("waits for the top-up card to close before claiming", async () => {
  const { claims } = setup();
  const { rerender } = render(tree(true));
  await new Promise((resolve) => setTimeout(resolve, 20));
  expect(claims).toEqual([]);
  rerender(tree(false));
  expect(await screen.findByText(QUESTION)).toBeDefined();
  expect(claims).toEqual([{ kind: "nps" }]);
});
