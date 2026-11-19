import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { focusManager } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { useWorkAreaStore } from "../../../workArea";
import { HomeBoard } from "../components/HomeBoard/HomeBoard";
import { makeTask } from "./taskFixtures";

let reads: number;
let tasks: ReturnType<typeof makeTask>[];

beforeEach(() => {
  reads = 0;
  tasks = [];
  useWorkAreaStore.getState().reset();
  server.use(
    getListExpertsMockHandler200([]),
    http.get("/api/proxy/api/tasks", () => {
      reads += 1;
      return HttpResponse.json({ tasks });
    }),
  );
});

afterEach(() => {
  focusManager.setFocused(undefined);
});

describe("Home Tasks — first read and coming back", () => {
  it("shows a skeleton on the first read, then the first steps when there is nothing yet", async () => {
    render(<HomeBoard />);

    expect(screen.getByRole("status", { name: "Loading tasks" })).toBeDefined();
    expect(
      await screen.findByRole("heading", { name: "No tasks yet" }),
    ).toBeDefined();
    expect(screen.queryByText("Nothing here")).toBeNull();
  });

  it("re-reads the board when the window gets focus back", async () => {
    tasks = [makeTask("t1", "Pick a venue", "TODO")];
    render(<HomeBoard />);
    await screen.findAllByTestId("board-card");
    tasks = [
      makeTask("t2", "Made in another tab", "TODO"),
      makeTask("t1", "Pick a venue", "TODO"),
    ];

    act(() => {
      focusManager.setFocused(false);
      focusManager.setFocused(true);
    });

    expect(
      await screen.findByRole("button", { name: /^Made in another tab/ }),
    ).toBeDefined();
    expect(reads).toBeGreaterThan(1);
  });
});
