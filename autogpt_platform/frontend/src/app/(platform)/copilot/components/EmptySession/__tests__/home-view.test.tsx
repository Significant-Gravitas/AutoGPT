import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListTasksMockHandler200 } from "@/app/api/__generated__/endpoints/tasks/tasks.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { makeSession, makeTask } from "./taskFixtures";

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

function renderHome(searchParams = "") {
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
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

beforeEach(() => {
  server.use(
    getListExpertIdentitiesMockHandler([]),
    getListExpertsMockHandler200([]),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
    getListTasksMockHandler200({
      tasks: [
        makeTask("task-1", "Pick a venue for the offsite", "NEEDS_INPUT"),
        makeTask("task-2", "Draft the launch email", "IN_PROGRESS"),
      ],
    }),
  );
});

describe("Home — Chat / Board switch", () => {
  it("opens on Chat, and Board swaps the composer for the task board", async () => {
    const user = userEvent.setup();
    renderHome();

    const views = await screen.findByRole("radiogroup", { name: "Home view" });
    expect(
      within(views)
        .getByRole("radio", { name: "Chat" })
        .getAttribute("aria-checked"),
    ).toBe("true");
    expect(
      await screen.findByPlaceholderText(/What's your role/),
    ).toBeDefined();

    await user.click(within(views).getByRole("radio", { name: "Board" }));

    expect(
      await screen.findByRole("heading", { name: "Needs you" }),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: "Working" })).toBeDefined();
    expect(
      await screen.findByText("Pick a venue for the offsite"),
    ).toBeDefined();
    expect(screen.getByText("Draft the launch email")).toBeDefined();
    expect(screen.queryByPlaceholderText(/What's your role/)).toBeNull();

    await user.click(screen.getByRole("radio", { name: "Chat" }));

    expect(
      await screen.findByPlaceholderText(/What's your role/),
    ).toBeDefined();
    expect(screen.queryByRole("heading", { name: "Needs you" })).toBeNull();
  });

  it("opens straight on the board from the link", async () => {
    renderHome("?show=board");

    expect(
      await screen.findByText("Pick a venue for the offsite"),
    ).toBeDefined();
    expect(
      screen.getByRole("radio", { name: "Board" }).getAttribute("aria-checked"),
    ).toBe("true");
  });
});

describe("Home — recent chats", () => {
  it("lists the recipient's latest chats under the composer, each with its state", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          makeSession("session-a", "Weekly report", null, {
            is_processing: true,
          }),
          makeSession("session-b", "Plan the offsite", null),
          makeSession("session-c", "Launch plan", "expert-maria"),
        ],
        total: 3,
      }),
    );
    renderHome();

    const prompt = await screen.findByPlaceholderText(/What's your role/);
    const recent = await screen.findByRole("region", { name: "Recent chats" });
    expect(
      prompt.compareDocumentPosition(recent) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    const links = within(recent).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "/home?sessionId=session-a",
      "/home?sessionId=session-b",
    ]);
    expect(links[0].textContent).toContain("Working: Weekly report");
    expect(links[1].textContent).toContain("Done: Plan the offsite");
    expect(within(recent).queryByText("Launch plan")).toBeNull();
  });

  it("shows no recent chats block when the recipient has none", async () => {
    renderHome();

    await screen.findByPlaceholderText(/What's your role/);
    expect(screen.queryByRole("region", { name: "Recent chats" })).toBeNull();
  });
});
