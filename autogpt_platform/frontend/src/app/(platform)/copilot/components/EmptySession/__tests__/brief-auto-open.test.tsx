import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getListTaskSuggestionsMockHandler200,
  getListTasksMockHandler200,
} from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import type { Task } from "@/app/api/__generated__/models/task";
import { server } from "@/mocks/mock-server";
import { Key, storage } from "@/services/storage/local-storage";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { BRIEF_TAB_KEY, useWorkAreaStore } from "../../../workArea";
import { EmptySession } from "../EmptySession";
import { makeTask } from "./taskFixtures";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

function drafted(id: string, title: string): Task {
  return { ...makeTask(id, title, "TODO"), is_suggested: true };
}

function renderHome() {
  const Wrapper = withNuqsTestingAdapter({ searchParams: "", hasMemory: true });
  return render(
    <Wrapper>
      <EmptySession
        isCreatingSession={false}
        onCreateSession={() => {}}
        onSend={vi.fn()}
      />
    </Wrapper>,
  );
}

let homeReads = 0;

/** Until Home has read the dashboard after `reads` earlier reads, and the
 *  effects that follow it have run. */
async function homeLoaded(reads: number) {
  await waitFor(() => expect(homeReads).toBeGreaterThan(reads));
  await new Promise((resolve) => setTimeout(resolve, 50));
}

function serve(suggestions: Task[]) {
  homeReads = 0;
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getListExpertsMockHandler200([]),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () => {
      homeReads += 1;
      return HttpResponse.json(makeDashboard([]));
    }),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getListTasksMockHandler200({ tasks: [] }),
    getListTaskSuggestionsMockHandler200({ tasks: suggestions }),
  );
}

beforeEach(() => {
  window.localStorage.clear();
  useWorkAreaStore.getState().reset();
});

describe("Home — the brief starts itself", () => {
  it("opens the brief on the first visit of the day, and not again that day", async () => {
    serve([]);
    const first = renderHome();

    await waitFor(() =>
      expect(useWorkAreaStore.getState().activeKey).toBe(BRIEF_TAB_KEY),
    );
    expect(storage.get(Key.HOME_BRIEF_AUTO_OPENED)).toBe(
      new Date().toDateString(),
    );

    // The user closes it, then comes back to Home the same day.
    useWorkAreaStore.getState().close(BRIEF_TAB_KEY);
    first.unmount();
    const reads = homeReads;
    renderHome();

    await homeLoaded(reads);
    expect(useWorkAreaStore.getState().tabs).toEqual([]);
  });

  it("opens again on a new day", async () => {
    storage.set(Key.HOME_BRIEF_AUTO_OPENED, "Thu Nov 13 2026");
    serve([]);
    renderHome();

    await waitFor(() =>
      expect(useWorkAreaStore.getState().activeKey).toBe(BRIEF_TAB_KEY),
    );
  });
});

describe("Home — drafted tasks line", () => {
  it("says how many tasks Otto drafted and opens the brief", async () => {
    storage.set(Key.HOME_BRIEF_AUTO_OPENED, new Date().toDateString());
    serve([drafted("s-1", "Pricing deck"), drafted("s-2", "Vendor calls")]);
    const user = userEvent.setup();
    renderHome();

    await user.click(
      await screen.findByRole("button", {
        name: "Otto drafted 2 tasks for you",
      }),
    );

    expect(useWorkAreaStore.getState().activeKey).toBe(BRIEF_TAB_KEY);
    expect(
      screen.queryByRole("button", { name: "Otto drafted 2 tasks for you" }),
    ).toBeNull();
  });
});
