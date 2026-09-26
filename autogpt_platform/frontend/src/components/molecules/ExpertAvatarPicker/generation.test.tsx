import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

// The hook polls on a real 2s interval, so the default 5s budget is not enough
// on a loaded CI runner.
const POLLING_TEST_TIMEOUT = 20000;

test(
  "polls a pending job until the preview is ready",
  async () => {
    let polls = 0;
    server.use(
      http.post("*/api/experts/avatars/generations", () =>
        HttpResponse.json(
          { id: "poll-job", status: "pending" },
          { status: 202 },
        ),
      ),
      http.get("*/api/experts/avatars/generations/poll-job", () => {
        polls += 1;
        return HttpResponse.json(
          polls === 1
            ? { id: "poll-job", status: "pending" }
            : {
                id: "poll-job",
                status: "complete",
                avatar_url: "https://cdn.test/sculpted.png",
              },
        );
      }),
    );
    render(
      <ExpertAvatarPicker
        name="Nova"
        category="development"
        autoGenerate
        onPick={vi.fn()}
      />,
    );

    await waitFor(() => expect(polls).toBe(1));
    expect(screen.queryByRole("status")).not.toBeNull();
    await waitFor(
      () =>
        expect(
          screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
        ).toContain("sculpted.png"),
      { timeout: 10000 },
    );
    expect(polls).toBe(2);
    expect(screen.queryByRole("status")).toBeNull();
  },
  POLLING_TEST_TIMEOUT,
);
