import { screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";

import { server } from "@/mocks/mock-server";
import {
  captureException,
  memoryChatTestConfig,
  renderMemoryPage,
  setupMemoryChat,
} from "./chat-test-helpers";

beforeEach(setupMemoryChat);

async function openSummaryChat() {
  renderMemoryPage();
  await userEvent
    .setup()
    .click(await screen.findByRole("button", { name: "View my summary" }));
}

describe("Memory chat start", () => {
  it.each([
    [
      "a subscription error",
      () =>
        HttpResponse.json({ detail: "Upgrade to continue." }, { status: 402 }),
    ],
    ["a null body", () => HttpResponse.json(null, { status: 402 })],
    [
      "structured detail",
      () =>
        HttpResponse.json(
          { detail: { message: "<script>unexpected()</script>" } },
          { status: 402 },
        ),
    ],
    [
      "invalid JSON",
      () =>
        new HttpResponse("not json", {
          status: 402,
          headers: { "Content-Type": "application/json" },
        }),
    ],
  ])(
    "offers plans for a 402 with %s without reporting an exception",
    async (_case, response) => {
      server.use(http.post("*/api/chat/sessions", response));

      await openSummaryChat();

      expect(
        await screen.findByText("The memory chat needs an active plan"),
      ).toBeDefined();
      expect(screen.getByText("Choose one to continue.")).toBeDefined();
      const plans = screen.getByRole("link", { name: "See plans" });
      expect(plans.getAttribute("href")).toBe("/settings/billing");
      expect(
        screen.queryByRole("complementary", { name: "Memory chat panel" }),
      ).toBeNull();
      expect(screen.queryByText("Could not start the memory chat")).toBeNull();
      expect(captureException).not.toHaveBeenCalled();

      await userEvent.setup().click(plans);
      await waitFor(() =>
        expect(
          screen.queryByText("The memory chat needs an active plan"),
        ).toBeNull(),
      );
    },
  );

  it("provides administrator guidance when billing is disabled", async () => {
    memoryChatTestConfig.billingEnabled = false;
    server.use(
      http.post("*/api/chat/sessions", () =>
        HttpResponse.json({}, { status: 402 }),
      ),
    );

    await openSummaryChat();

    expect(
      await screen.findByText("Ask whoever runs this server to set one up."),
    ).toBeDefined();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(
      screen.queryByRole("complementary", { name: "Memory chat panel" }),
    ).toBeNull();
    expect(captureException).not.toHaveBeenCalled();
  });

  it.each([401, 403, 404, 409, 422, 429, 500, 503])(
    "keeps HTTP %s retryable even when its body mentions a paywall",
    async (status) => {
      server.use(
        http.post("*/api/chat/sessions", () =>
          HttpResponse.json(
            {
              status: 402,
              detail: "A subscription is required. Upgrade to continue.",
            },
            { status },
          ),
        ),
      );

      await openSummaryChat();

      expect(
        await screen.findByText("Could not start the memory chat"),
      ).toBeDefined();
      expect(screen.getByRole("button", { name: "Retry" })).toBeDefined();
      expect(captureException).toHaveBeenCalledWith(
        expect.objectContaining({ status }),
      );
      expect(
        screen.queryByText("The memory chat needs an active plan"),
      ).toBeNull();
      expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    },
  );

  it("keeps network failures retryable without offering a plan", async () => {
    server.use(http.post("*/api/chat/sessions", () => HttpResponse.error()));

    await openSummaryChat();

    expect(
      await screen.findByText("Could not start the memory chat"),
    ).toBeDefined();
    expect(captureException).toHaveBeenCalledWith(expect.any(Error));
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
  });

  it("does not mistake a malformed success body for a paywall", async () => {
    server.use(
      http.post("*/api/chat/sessions", () =>
        HttpResponse.json({ status: 402 }),
      ),
    );

    await openSummaryChat();

    expect(
      await screen.findByText("Could not start the memory chat"),
    ).toBeDefined();
    expect(captureException).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "Failed to create memory chat session",
      }),
    );
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
  });

  it("retries a temporary subscription lookup failure and starts the chat", async () => {
    server.use(
      http.post(
        "*/api/chat/sessions",
        () =>
          HttpResponse.json(
            {
              detail:
                "Subscription state temporarily unavailable, retry shortly.",
            },
            { status: 503 },
          ),
        { once: true },
      ),
    );

    await openSummaryChat();
    await userEvent
      .setup()
      .click(await screen.findByRole("button", { name: "Retry" }));

    expect(await screen.findByText("Your memory summary.")).toBeDefined();
    expect(screen.queryByText("Could not start the memory chat")).toBeNull();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
  });
});
