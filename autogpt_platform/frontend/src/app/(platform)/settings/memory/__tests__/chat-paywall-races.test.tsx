import { screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, describe, expect, it } from "vitest";

import { server } from "@/mocks/mock-server";
import {
  captureException,
  renderMemoryPage,
  setupMemoryChat,
  waitForSessionCreates,
} from "./chat-test-helpers";

beforeEach(setupMemoryChat);

describe("Memory chat request lifetime", () => {
  it.each([402, 503])(
    "ignores a late HTTP %s from a previous scope",
    async (status) => {
      const response = Promise.withResolvers<Response>();
      server.use(
        http.post("*/api/chat/sessions", () => response.promise, {
          once: true,
        }),
      );
      renderMemoryPage();
      const user = userEvent.setup();
      await user.click(
        await screen.findByRole("button", { name: "View my summary" }),
      );
      await waitForSessionCreates(1);

      await user.click(screen.getByRole("button", { name: "Memory scope" }));
      await user.click(
        await screen.findByRole("menuitem", { name: /Growth Marketer/ }),
      );
      await user.click(
        await screen.findByRole("button", { name: "View Maria's summary" }),
      );
      expect(await screen.findByText("Your memory summary.")).toBeDefined();

      response.resolve(
        HttpResponse.json({ detail: "Upgrade to continue." }, { status }),
      );
      await waitForSessionCreates(0);

      const panel = screen.getByRole("complementary", {
        name: "Memory chat panel",
      });
      expect(within(panel).getByText("Maria's memory")).toBeDefined();
      expect(within(panel).getByText("Your memory summary.")).toBeDefined();
      expect(screen.queryByText("Could not start the memory chat")).toBeNull();
      expect(
        screen.queryByText("The memory chat needs an active plan"),
      ).toBeNull();
      if (status === 402) expect(captureException).not.toHaveBeenCalled();
    },
  );

  it("ignores a late paywall response after leaving memory settings", async () => {
    const response = Promise.withResolvers<Response>();
    server.use(http.post("*/api/chat/sessions", () => response.promise));
    renderMemoryPage();
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "View my summary" }),
    );
    await waitForSessionCreates(1);
    await user.click(
      screen.getByRole("button", { name: "Leave memory settings" }),
    );

    response.resolve(HttpResponse.json({}, { status: 402 }));
    await waitForSessionCreates(0);

    expect(
      screen.queryByText("The memory chat needs an active plan"),
    ).toBeNull();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(captureException).not.toHaveBeenCalled();
  });

  it("keeps a newer scope's create locked and handles its own paywall", async () => {
    const oldResponse = Promise.withResolvers<Response>();
    const newResponse = Promise.withResolvers<Response>();
    const bodies: unknown[] = [];
    server.use(
      http.post("*/api/chat/sessions", async ({ request }) => {
        bodies.push(await request.json());
        return bodies.length === 1 ? oldResponse.promise : newResponse.promise;
      }),
    );
    renderMemoryPage();
    const user = userEvent.setup();
    await user.click(
      await screen.findByRole("button", { name: "View my summary" }),
    );
    await waitForSessionCreates(1);
    await user.click(screen.getByRole("button", { name: "Memory scope" }));
    await user.click(
      await screen.findByRole("menuitem", { name: /Growth Marketer/ }),
    );
    await user.click(
      await screen.findByRole("button", { name: "View Maria's summary" }),
    );
    await waitForSessionCreates(2);

    oldResponse.resolve(HttpResponse.json({}, { status: 402 }));
    await waitForSessionCreates(1);
    expect(
      screen.queryByText("The memory chat needs an active plan"),
    ).toBeNull();
    await user.click(
      screen.getByRole("button", { name: "View Maria's summary" }),
    );
    await waitForSessionCreates(1);
    expect(bodies).toEqual([null, { expert_id: "expert-maria" }]);

    newResponse.resolve(HttpResponse.json({}, { status: 402 }));
    await waitForSessionCreates(0);
    expect(
      await screen.findByRole("link", { name: "See plans" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("complementary", { name: "Memory chat panel" }),
    ).toBeNull();
    expect(captureException).not.toHaveBeenCalled();
  });
});
