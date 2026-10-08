import { http, HttpResponse } from "msw";
import { expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { attentionFixture } from "./fixtures";
import { MobileWorkspace } from "../components/MobileWorkspace";

vi.mock("@/services/feature-flags/use-get-flag", async (original) => ({
  ...(await original<object>()),
  useGetFlag: () => true,
}));
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

it("opens an existing chat without replacing its session or expert", async () => {
  server.use(
    http.get("*/api/chat/sessions", () =>
      HttpResponse.json({
        sessions: [
          {
            id: "session-1",
            title: "Launch plan",
            expert_id: "maria",
            is_processing: false,
            created_at: "2026-10-07",
            updated_at: "2026-10-07",
          },
        ],
        total: 1,
      }),
    ),
  );
  render(<MobileWorkspace tab="chats" />);
  expect(
    (await screen.findByRole("link", { name: /Launch plan/ })).getAttribute(
      "href",
    ),
  ).toBe("/home?sessionId=session-1");
});

it("lets the user start an expert conversation and excludes archived experts", async () => {
  server.use(
    http.get("*/api/experts/identities", () =>
      HttpResponse.json([
        { id: "maria", name: "Maria", role: "Marketing", is_archived: false },
        { id: "old", name: "Archived expert", role: "Old", is_archived: true },
      ]),
    ),
  );
  render(<MobileWorkspace tab="experts" />);
  expect(
    (await screen.findByRole("link", { name: /Chat with Maria/ })).getAttribute(
      "href",
    ),
  ).toBe("/home?expertId=maria");
  expect(screen.queryByRole("link", { name: /Archived expert/ })).toBeNull();
});

it("keeps a failed attention fetch distinct from an empty prompt inbox", async () => {
  server.use(
    http.get("*/api/home", () => new HttpResponse(null, { status: 503 })),
  );
  render(<MobileWorkspace tab="attention" />);
  expect(
    await screen.findByRole("button", { name: /try again/i }),
  ).not.toBeNull();
  expect(screen.queryByText("You're all caught up.")).toBeNull();
});

it("opens questions in their conversation and submits explicit approval for the selected request", async () => {
  const decisions: unknown[] = [];
  server.use(
    http.get("*/api/home", () =>
      HttpResponse.json({ attention: attentionFixture }),
    ),
    http.post("*/api/review/action", async ({ request }) => {
      decisions.push(await request.json());
      return HttpResponse.json({
        processed_count: 1,
        failed_count: 0,
        error: null,
      });
    }),
  );
  render(<MobileWorkspace tab="attention" />);
  expect(
    (await screen.findByRole("link", { name: "Answer" })).getAttribute("href"),
  ).toBe("/home?sessionId=session-1");
  expect(decisions).toHaveLength(0);
  await userEvent
    .setup()
    .click(
      screen.getByRole("button", { name: "Approve: Review the launch email" }),
    );
  await waitFor(() =>
    expect(decisions).toEqual([
      {
        reviews: [
          {
            node_exec_id: "node-1",
            approved: true,
            auto_approve_future: false,
          },
        ],
      },
    ]),
  );
});
