import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListTasksMockHandler200 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { act, render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { taskTabKey, useWorkAreaStore } from "../../../workArea";
import { HomeBoard } from "../components/HomeBoard/HomeBoard";
import { makeTask } from "./taskFixtures";

let onUrlUpdate: ReturnType<typeof vi.fn<(event: UrlUpdateEvent) => void>>;

function linkedTask() {
  return onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("task");
}

function renderBoard(searchParams = "?show=tasks") {
  render(
    <NuqsTestingAdapter searchParams={searchParams} onUrlUpdate={onUrlUpdate}>
      <HomeBoard />
    </NuqsTestingAdapter>,
  );
}

beforeEach(() => {
  onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
  useWorkAreaStore.getState().reset();
  server.use(
    getListExpertsMockHandler200([]),
    getListTasksMockHandler200({
      tasks: [
        makeTask("task-1", "Pick a venue for the offsite", "NEEDS_INPUT"),
        makeTask("task-2", "Draft the launch email", "IN_PROGRESS"),
      ],
    }),
  );
});

describe("Home Tasks — the open task in the address", () => {
  it("writes the task it opens and clears it when its tab closes", async () => {
    renderBoard();

    await userEvent.click(
      await screen.findByRole("button", { name: /^Pick a venue/ }),
    );
    await vi.waitFor(() => expect(linkedTask()).toBe("task-1"));

    await userEvent.click(
      screen.getByRole("button", { name: /^Draft the launch/ }),
    );
    await vi.waitFor(() => expect(linkedTask()).toBe("task-2"));

    act(() => useWorkAreaStore.getState().close(taskTabKey("task-2")));
    await vi.waitFor(() => expect(linkedTask()).toBe("task-1"));

    act(() => useWorkAreaStore.getState().close(taskTabKey("task-1")));
    await vi.waitFor(() => expect(linkedTask()).toBeNull());
  });

  it("keeps a shared link until its task has opened and closed", async () => {
    renderBoard("?show=tasks&task=task-2");
    await screen.findAllByTestId("board-card");

    expect(onUrlUpdate).not.toHaveBeenCalled();

    act(() => useWorkAreaStore.getState().openTask("task-2", "Task"));
    act(() => useWorkAreaStore.getState().close(taskTabKey("task-2")));

    await vi.waitFor(() => expect(linkedTask()).toBeNull());
  });
});
