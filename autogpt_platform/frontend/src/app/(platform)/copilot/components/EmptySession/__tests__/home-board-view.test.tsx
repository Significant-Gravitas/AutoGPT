import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListTasksMockHandler200 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { HomeBoard } from "../components/HomeBoard/HomeBoard";
import { makeTask } from "./taskFixtures";

function pressed(name: string) {
  return screen.getByRole("button", { name }).getAttribute("aria-pressed");
}

beforeEach(() => {
  storage.clean(Key.HOME_TASKS_VIEW);
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

describe("Home Tasks — Board / List switch", () => {
  it("opens on the board, flips to the list, and remembers the list", async () => {
    const user = userEvent.setup();
    const first = render(<HomeBoard />);

    expect(await screen.findAllByTestId("board-card")).toHaveLength(2);
    expect(pressed("Board view")).toBe("true");
    expect(pressed("List view")).toBe("false");

    await user.click(screen.getByRole("button", { name: "List view" }));

    expect(await screen.findAllByTestId("list-row")).toHaveLength(2);
    expect(screen.queryAllByTestId("board-card")).toHaveLength(0);
    expect(pressed("List view")).toBe("true");
    first.unmount();

    render(<HomeBoard />);

    expect(await screen.findAllByTestId("list-row")).toHaveLength(2);
    expect(pressed("List view")).toBe("true");

    await user.click(screen.getByRole("button", { name: "Board view" }));

    expect(await screen.findAllByTestId("board-card")).toHaveLength(2);
    expect(storage.get(Key.HOME_TASKS_VIEW)).toBe("board");
  });
});
