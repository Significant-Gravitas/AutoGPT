import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getGetV2GetALinkPurchaseApprovalMockHandler,
  getPostV2ApproveALinkPurchaseMockHandler,
  getPostV2DeclineALinkPurchaseMockHandler,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { LinkPurchaseApproval } from "@/app/api/__generated__/models/linkPurchaseApproval";
import { CopilotChatActionsProvider } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { LinkCheckout } from "../LinkCheckout";

const SESSION_ID = "chat-session";
const CHECKOUT_ID = "c".repeat(32);
const onSend = vi.fn();

const output = {
  type: "browser_checkout",
  session_id: SESSION_ID,
  checkout_id: CHECKOUT_ID,
  merchant_name: "Test bookshop",
  amount: 1299,
  currency: "usd",
  test_mode: true,
  status: "awaiting_approval",
  approval_mode: "in_app",
  message: "The purchase is shown in the chat for the user to approve.",
};

function approval(
  state: LinkPurchaseApproval["state"],
  changes: Partial<LinkPurchaseApproval> = {},
): LinkPurchaseApproval {
  return {
    checkout_id: CHECKOUT_ID,
    state,
    merchant_name: "Test bookshop",
    merchant_url: "https://books.example/checkout",
    context: "One paperback from the signed-in cart, as asked in this chat.",
    amount: 1299,
    currency: "usd",
    test_mode: true,
    expires_at: Date.now() / 1000 + 600,
    ...changes,
  };
}

function renderCard(chatSurface: "copilot" | "share" = "copilot") {
  return render(
    <CopilotChatActionsProvider onSend={onSend} chatSurface={chatSurface}>
      <LinkCheckout output={output} />
    </CopilotChatActionsProvider>,
  );
}

describe("LinkCheckout in-chat approval", () => {
  beforeEach(() => {
    onSend.mockClear();
  });

  it("approves in the chat, then lets the agent complete the purchase", async () => {
    let state: LinkPurchaseApproval["state"] = "awaiting";
    const approve = vi.fn();
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval(state)),
      getPostV2ApproveALinkPurchaseMockHandler(() => {
        approve();
        state = "approved";
        return approval(state);
      }),
    );
    renderCard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Approve $12.99" }),
    );

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(approve).toHaveBeenCalledTimes(1);
    expect(onSend.mock.calls[0][0]).toContain(
      `run_capability id "tool:browser_complete_link_payment" and input {"checkout_id":"${CHECKOUT_ID}"}`,
    );
    expect(
      await screen.findByText("Approved. Completing the purchase…"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();
  });

  it("declines without paying and tells the agent not to buy", async () => {
    let state: LinkPurchaseApproval["state"] = "awaiting";
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval(state)),
      getPostV2DeclineALinkPurchaseMockHandler(() => {
        state = "declined";
        return approval(state);
      }),
    );
    renderCard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Decline" }),
    );

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1));
    expect(onSend.mock.calls[0][0]).toContain("I declined purchase");
    // The agent wrote the merchant name; it is not repeated back as the user.
    expect(onSend.mock.calls[0][0]).not.toContain("Test bookshop");
    expect(
      await screen.findByText("Declined. Nothing was charged."),
    ).toBeDefined();
  });

  it("shows where a purchase stands when the decision was not recorded", async () => {
    let state: LinkPurchaseApproval["state"] = "awaiting";
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval(state)),
      http.post("*/link-checkouts/:checkoutId/approve", () => {
        state = "declined";
        return HttpResponse.json(
          { detail: "This purchase was already declined" },
          { status: 409 },
        );
      }),
    );
    renderCard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Approve $12.99" }),
    );

    expect(
      await screen.findByText("Declined. Nothing was charged."),
    ).toBeDefined();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("shows the purchase as the server recorded it, site included", async () => {
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() =>
        approval("awaiting", {
          merchant_url: "https://pay.elsewhere.example/checkout",
          amount: 4999,
        }),
      ),
    );
    renderCard();

    expect(
      await screen.findByRole("button", { name: "Approve $49.99" }),
    ).toBeDefined();
    expect(screen.getByText("pay.elsewhere.example")).toBeDefined();
    expect(
      screen.getByText(
        "One paperback from the signed-in cart, as asked in this chat.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Approve $12.99" })).toBeNull();
  });

  it("keeps the transcript's total out of the header while approving", async () => {
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() =>
        approval("awaiting", { amount: 4999 }),
      ),
    );
    renderCard();

    expect(
      await screen.findByRole("button", { name: "Approve $49.99" }),
    ).toBeDefined();
    expect(screen.queryByText("$12.99")).toBeNull();
  });

  it("offers a retry when the purchase cannot be loaded", async () => {
    let fail = true;
    server.use(
      http.get("*/link-checkouts/:checkoutId", () =>
        fail
          ? HttpResponse.json({ detail: "unavailable" }, { status: 500 })
          : HttpResponse.json(approval("awaiting")),
      ),
    );
    renderCard();

    expect(
      await screen.findByText("This purchase could not be loaded."),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();

    fail = false;
    await userEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(
      await screen.findByRole("button", { name: "Approve $12.99" }),
    ).toBeDefined();
  });

  it("reads a purchase that no longer exists as no longer waiting", async () => {
    server.use(
      http.get("*/link-checkouts/:checkoutId", () =>
        HttpResponse.json({ detail: "Purchase not found" }, { status: 404 }),
      ),
    );
    renderCard();

    expect(
      await screen.findByText(
        "This purchase is no longer waiting for approval.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  it("lets the user resend when the agent was not told", async () => {
    let state: LinkPurchaseApproval["state"] = "awaiting";
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval(state)),
      getPostV2ApproveALinkPurchaseMockHandler(() => {
        state = "approved";
        return approval(state);
      }),
    );
    onSend.mockRejectedValueOnce(new Error("offline"));
    renderCard();

    await userEvent.click(
      await screen.findByRole("button", { name: "Approve $12.99" }),
    );

    expect(
      await screen.findByText(
        "Your decision is saved, but the message to the agent did not send.",
      ),
    ).toBeDefined();
    await userEvent.click(
      screen.getByRole("button", { name: "Tell the agent" }),
    );

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(2));
    expect(onSend.mock.calls[1][0]).toContain(`"checkout_id":"${CHECKOUT_ID}"`);
    await waitFor(() =>
      expect(
        screen.queryByRole("button", { name: "Tell the agent" }),
      ).toBeNull(),
    );
  });

  it("shows a decision already made after the chat reloads", async () => {
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval("approved")),
    );
    renderCard();

    expect(
      await screen.findByText("Approved. Completing the purchase…"),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();
  });

  it("never offers the decision on a shared conversation", async () => {
    server.use(
      getGetV2GetALinkPurchaseApprovalMockHandler(() => approval("awaiting")),
    );
    renderCard("share");

    expect(
      await screen.findByText(/The agent never sees your card number/),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: /Approve/ })).toBeNull();
    expect(screen.queryByRole("button", { name: "Decline" })).toBeNull();
  });
});
