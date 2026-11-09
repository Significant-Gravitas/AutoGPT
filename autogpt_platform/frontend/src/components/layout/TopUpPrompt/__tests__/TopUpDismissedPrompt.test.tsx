import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import {
  getGetV1GetAutoTopUpMockHandler,
  getGetV1GetUserCreditsMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { TopUpDialog } from "@/components/layout/TopUpPrompt/TopUpDialog/TopUpDialog";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { useTopUpPrompt } from "@/components/layout/TopUpPrompt/useTopUpPrompt";
import {
  resetCheckoutRedirect,
  useCheckoutRedirect,
} from "@/hooks/useCheckoutRedirect";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { id: "user-a" }, isLoggedIn: true }),
}));

vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: false, isResolved: true }),
}));

const CARD_TITLE = "What would make topping up worth it?";

function Controls() {
  const { openTopUp, isTopUpCardOpen } = useTopUpPrompt();
  const { markRedirecting } = useCheckoutRedirect();
  return (
    <>
      <button onClick={openTopUp}>Open top-up</button>
      <button onClick={markRedirecting}>Leave for Stripe</button>
      <output>{isTopUpCardOpen ? "card open" : "card closed"}</output>
    </>
  );
}

function setup(claimedId: string | null = "prompt-1") {
  const claims: unknown[] = [];
  const patches: unknown[] = [];
  server.use(
    getGetV1GetUserCreditsMockHandler({ credits: 100 }),
    getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }),
    getPostV2ClaimFeedbackPromptMockHandler200(async ({ request }) => {
      claims.push(await request.clone().json());
      return { id: claimedId };
    }),
    getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
      patches.push(await request.clone().json());
      return {
        id: "prompt-1",
        kind: "top_up",
        answered: true,
        dismissed: false,
        reason: null,
        score: null,
      };
    }),
  );
  render(
    <TopUpPromptProvider>
      <Controls />
    </TopUpPromptProvider>,
  );
  return { claims, patches };
}

async function closeTopUpDialog() {
  fireEvent.click(screen.getByRole("button", { name: "Open top-up" }));
  await screen.findByText("You're out of automation credits");
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  await waitFor(() =>
    expect(screen.queryByText("You're out of automation credits")).toBeNull(),
  );
}

beforeEach(() => vi.clearAllMocks());
afterEach(() => resetCheckoutRedirect());

test("closing the out-of-credits dialog claims and shows the card", async () => {
  const { claims } = setup();
  await closeTopUpDialog();
  expect(await screen.findByText(CARD_TITLE)).toBeDefined();
  expect(claims).toEqual([{ kind: "top_up" }]);
});

test("closing the add-credits dialog reports nothing", () => {
  const onDismissed = vi.fn();
  render(
    <TopUpDialog
      isOpen
      onClose={() => {}}
      variant="add-credits"
      onDismissed={onDismissed}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  expect(onDismissed).not.toHaveBeenCalled();
});

test("closing on the way to Stripe shows nothing", async () => {
  const { claims } = setup();
  fireEvent.click(screen.getByRole("button", { name: "Leave for Stripe" }));
  await closeTopUpDialog();
  // Set by the close itself, so it is the guard this reads, not a slow claim.
  expect(screen.getByText("card closed")).toBeDefined();
  expect(screen.queryByText(CARD_TITLE)).toBeNull();
  expect(claims).toEqual([]);
});

test("a refused claim shows nothing and clears the card state", async () => {
  setup(null);
  await closeTopUpDialog();
  await waitFor(() => expect(screen.getByText("card closed")).toBeDefined());
  expect(screen.queryByText(CARD_TITLE)).toBeNull();
});

test("picking a reason saves it", async () => {
  const { patches } = setup();
  await closeTopUpDialog();
  await screen.findByText(CARD_TITLE);
  fireEvent.click(screen.getByRole("button", { name: "Smaller amounts" }));
  await waitFor(() => expect(patches).toEqual([{ reason: "smaller_amounts" }]));
});

test("closing the card saves a dismissal and frees the card state", async () => {
  const { patches } = setup();
  await closeTopUpDialog();
  await screen.findByText(CARD_TITLE);
  fireEvent.click(screen.getByRole("button", { name: "Close" }));
  await waitFor(() => expect(patches).toEqual([{ dismissed: true }]));
  expect(screen.getByText("card closed")).toBeDefined();
});
