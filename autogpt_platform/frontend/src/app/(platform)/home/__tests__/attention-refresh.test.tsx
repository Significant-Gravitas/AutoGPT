import {
  focusManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { afterEach, describe, expect, test, vi } from "vitest";
import { getGetHomeDashboardMockHandler200 } from "@/app/api/__generated__/endpoints/home/home.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { ExpertNeedsYouSection } from "@/app/(platform)/team/[expertId]/components/ExpertNeedsYouSection/ExpertNeedsYouSection";
import { server } from "@/mocks/mock-server";
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import { HomeRecap } from "../components/HomeRecap/HomeRecap";
import { ada, makeDashboard } from "./heldItems";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
    useGetFlag: () => false,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { user_metadata: { preferred_name: "Sam" } } }),
}));

const question: HomeAttentionItem = {
  id: "question-first",
  kind: "question",
  priority: "normal",
  title: "Ada has a question",
  description: "Which day should I schedule the meeting?",
  why_it_matters: "The work is paused until you answer in the chat.",
  expert: ada,
  primary_action: { label: "Answer", href: "/copilot?sessionId=first" },
};
const otherQuestion: HomeAttentionItem = {
  ...question,
  id: "question-second",
  title: "Ada has another question",
  description: "Who should I invite?",
  primary_action: { label: "Answer", href: "/copilot?sessionId=second" },
};

const clients: QueryClient[] = [];

afterEach(() => {
  focusManager.setFocused(undefined);
  clients.splice(0).forEach((client) => client.clear());
});

function createClient() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  clients.push(client);
  return client;
}

function renderPage(page: "home" | "expert", client: QueryClient) {
  return render(
    <QueryClientProvider client={client}>
      {page === "home" ? (
        <HomeRecap />
      ) : (
        <ExpertNeedsYouSection expert={{ id: ada.id } as Expert} />
      )}
    </QueryClientProvider>,
  );
}

function expectCount(count: number) {
  const label = `${count} ${count === 1 ? "item needs" : "items need"} your attention`;
  expect(
    screen
      .getAllByRole("status")
      .some((status) =>
        [status.getAttribute("aria-label"), status.textContent].includes(label),
      ),
  ).toBe(true);
}

describe.each(["home", "expert"] as const)("%s attention freshness", (page) => {
  test.each([0, 1])(
    "refreshes the list on return with %i questions remaining inside the cache window",
    async (remainingCount) => {
      const client = createClient();
      let attention = [question, otherQuestion];
      server.use(
        getGetHomeDashboardMockHandler200(() => makeDashboard(attention)),
      );
      const view = renderPage(page, client);
      await screen.findByText(question.description);
      expectCount(2);

      view.unmount();
      attention = remainingCount ? [otherQuestion] : [];
      renderPage(page, client);

      await waitFor(() => {
        expect(screen.queryByText(question.description)).toBeNull();
      });
      if (remainingCount) {
        expect(screen.getByText(otherQuestion.description)).toBeDefined();
        expectCount(1);
      } else {
        expect(screen.queryByText(otherQuestion.description)).toBeNull();
        expect(screen.queryByText("Needs you")).toBeNull();
      }
    },
  );

  test("refreshes answered questions when the window regains focus", async () => {
    const client = createClient();
    let attention = [question, otherQuestion];
    server.use(
      getGetHomeDashboardMockHandler200(() => makeDashboard(attention)),
    );
    renderPage(page, client);
    await screen.findByText(question.description);
    expectCount(2);

    act(() => focusManager.setFocused(false));
    attention = [otherQuestion];
    act(() => focusManager.setFocused(true));

    await waitFor(() => {
      expect(screen.queryByText(question.description)).toBeNull();
    });
    expect(screen.getByText(otherQuestion.description)).toBeDefined();
    expectCount(1);
  });

  test("keeps a question actionable when returning without answering", async () => {
    const client = createClient();
    const fetchDashboard = vi.fn(() => makeDashboard([question]));
    server.use(getGetHomeDashboardMockHandler200(fetchDashboard));
    const view = renderPage(page, client);
    await screen.findByText(question.description);

    view.unmount();
    renderPage(page, client);

    await waitFor(() => expect(fetchDashboard).toHaveBeenCalledTimes(2));
    expect(screen.getByText(question.description)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Answer" }).getAttribute("href"),
    ).toBe(question.primary_action.href);
    expectCount(1);
  });
});
