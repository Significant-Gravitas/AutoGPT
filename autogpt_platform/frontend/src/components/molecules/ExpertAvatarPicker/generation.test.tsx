import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import { ExpertAvatarPicker } from "./ExpertAvatarPicker";

test("sends shape and expression choices, then polls until the preview is ready", async () => {
  const onPick = vi.fn();
  const requests: unknown[] = [];
  let polls = 0;
  server.use(
    http.post("*/api/experts/avatars/generations", async ({ request }) => {
      requests.push(await request.json());
      return HttpResponse.json(
        { id: "poll-job", status: "pending" },
        { status: 202 },
      );
    }),
    http.get("*/api/experts/avatars/generations/poll-job", () => {
      polls += 1;
      return HttpResponse.json(
        polls === 1
          ? { id: "poll-job", status: "pending" }
          : {
              id: "poll-job",
              status: "complete",
              avatar_url: "https://cdn.test/focused.png",
            },
      );
    }),
  );
  render(<ExpertAvatarPicker name="Nova" color={null} onPick={onPick} />);
  await userEvent.click(screen.getByRole("button", { name: "Charcoal" }));
  await userEvent.click(screen.getByRole("combobox", { name: "Shape" }));
  await userEvent.click(screen.getByRole("option", { name: "Dome" }));
  await userEvent.click(screen.getByRole("combobox", { name: "Expression" }));
  await userEvent.click(screen.getByRole("option", { name: "Focused" }));
  await userEvent.click(
    screen.getByRole("button", { name: "Generate with AI" }),
  );
  await waitFor(() => expect(polls).toBe(1));
  expect(screen.queryByRole("status")).not.toBeNull();
  expect(requests).toEqual([
    { category: "development", shape: "dome", expression: "focused" },
  ]);
  await waitFor(
    () =>
      expect(
        screen.getByRole("img", { name: "Nova" }).getAttribute("src"),
      ).toContain("focused.png"),
    { timeout: 4000 },
  );
  expect(polls).toBe(2);
  expect(onPick).not.toHaveBeenCalled();
  await userEvent.click(
    screen.getByRole("button", { name: "Use this avatar" }),
  );
  expect(onPick).toHaveBeenCalledWith(
    "https://cdn.test/focused.png",
    "blue-300",
  );
});
