import {
  getGetV2GetSessionMockHandler200,
  getGetV2ListSessionsMockHandler200,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { http, HttpResponse } from "msw";
import { onboardingCard, onboardingTurn } from "./onboardingFixtures";
import { ExpertChatDrawer } from "../ExpertChatDrawer";

const EXPERT_ID = "expert-zara";
const SESSION_ID = "session-zara";

describe("ExpertChatDrawer", () => {
  test("keeps answers if creating the new chat fails", async () => {
    const createSession = vi.fn(() =>
      HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
    );
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", () =>
        HttpResponse.json(onboardingCard()),
      ),
      http.post("/api/proxy/api/chat/sessions", createSession),
    );
    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    fireEvent.click(await screen.findByRole("radio", { name: "Pricing" }));
    fireEvent.click(screen.getByRole("button", { name: "Send answers" }));
    await waitFor(() => expect(createSession).toHaveBeenCalled());
    await waitFor(() =>
      expect(
        (
          screen.getByRole("button", {
            name: "Send answers",
          }) as HTMLButtonElement
        ).disabled,
      ).toBe(false),
    );
    expect(
      screen
        .getByRole("radio", { name: "Pricing" })
        .getAttribute("aria-checked"),
    ).toBe("true");
  });

  test("keeps settled setup out of new chats", async () => {
    const request = vi.fn(() => HttpResponse.json(null));
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", request),
    );
    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    await waitFor(() => expect(request).toHaveBeenCalled());
    expect(screen.getByText("What can I do for you?")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Skip" })).toBeNull();
  });

  test("does not request setup when the new chat has a prompt", async () => {
    const request = vi.fn(() => HttpResponse.json(onboardingCard()));
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", request),
    );
    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        resumeLatest={false}
        seedPrompt="Research this company"
        onClose={() => {}}
      />,
    );
    expect(await screen.findByText("What can I do for you?")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
  });

  test("shows unanswered setup questions in a new chat", async () => {
    server.use(
      http.get("/api/proxy/api/experts/:expertId/onboarding", () =>
        HttpResponse.json(onboardingCard()),
      ),
    );
    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        resumeLatest={false}
        onClose={() => {}}
      />,
    );
    expect(
      await screen.findByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
  });

  test("a hire's onboarding card is a live form, not a settled row", async () => {
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions: [
          {
            id: SESSION_ID,
            created_at: "2026-09-27T18:00:00Z",
            updated_at: "2026-09-27T18:01:00Z",
            is_processing: false,
            expert_id: EXPERT_ID,
          },
        ],
        total: 1,
      }),
      getGetV2GetSessionMockHandler200({
        id: SESSION_ID,
        created_at: "2026-09-27T18:00:00Z",
        updated_at: "2026-09-27T18:01:00Z",
        user_id: "user-1",
        expert_id: EXPERT_ID,
        messages: onboardingTurn(),
      }),
    );

    render(
      <ExpertChatDrawer
        target={{
          expertId: EXPERT_ID,
          name: "Zara",
          role: "GTM Strategist",
          avatarUrl: null,
        }}
        onClose={() => {}}
      />,
    );

    expect(
      await screen.findByText("Which outcome should I start with?"),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Skip" })).toBeDefined();
    expect(screen.queryByText(/Setup questions from/)).toBeNull();
  });
});
