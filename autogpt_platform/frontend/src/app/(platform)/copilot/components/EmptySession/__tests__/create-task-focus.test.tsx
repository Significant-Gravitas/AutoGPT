import { getListExpertsMockHandler200 } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListTasksMockHandler200 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useWorkAreaStore } from "../../../workArea";
import { HomeBoard } from "../components/HomeBoard/HomeBoard";
import { makeTask } from "./taskFixtures";

beforeEach(() => {
  window.sessionStorage.clear();
  useWorkAreaStore.getState().reset();
  server.use(
    getListExpertsMockHandler200([]),
    getListTasksMockHandler200({
      tasks: [makeTask("t1", "Pick a venue", "NEEDS_INPUT")],
    }),
  );
});

async function closed() {
  await vi.waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "New task" })).toBeNull(),
  );
}

describe("Tasks view keyboard", () => {
  it("puts focus back on Create task when the dialog closes with Escape", async () => {
    const user = userEvent.setup();
    render(<HomeBoard />);
    await screen.findAllByTestId("board-card");
    const opener = screen.getByRole("button", { name: "Create task" });

    opener.focus();
    await user.keyboard("{Enter}");
    expect(
      await screen.findByRole("dialog", { name: "New task" }),
    ).toBeDefined();
    await user.keyboard("{Escape}");

    await closed();
    await vi.waitFor(() => expect(document.activeElement).toBe(opener));
  });

  it("opens Create task with N and goes to search with /, but not while typing", async () => {
    const user = userEvent.setup();
    render(<HomeBoard />);
    await screen.findAllByTestId("board-card");

    await user.keyboard("/");
    const search = screen.getByRole("searchbox", { name: "Search tasks" });
    expect(document.activeElement).toBe(search);

    await user.keyboard("n");
    expect((search as HTMLInputElement).value).toBe("n");
    expect(screen.queryByRole("dialog", { name: "New task" })).toBeNull();

    await user.clear(search);
    search.blur();
    await user.keyboard("n");
    expect(
      await screen.findByRole("dialog", { name: "New task" }),
    ).toBeDefined();
  });

  it("skips to the tasks from the top of the view", async () => {
    const user = userEvent.setup();
    render(<HomeBoard />);
    await screen.findAllByTestId("board-card");

    await user.tab();
    const skip = screen.getByRole("link", { name: "Skip to tasks" });
    expect(document.activeElement).toBe(skip);
    await user.keyboard("{Enter}");

    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /^Pick a venue/ }),
    );
  });
});
