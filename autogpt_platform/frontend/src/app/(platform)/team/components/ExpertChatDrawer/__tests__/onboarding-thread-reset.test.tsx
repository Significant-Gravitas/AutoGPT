import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { ExpertChatDrawer } from "../ExpertChatDrawer";
import { onboardingCard } from "./onboardingFixtures";

test("an old chat creation cannot settle setup in a new thread", async () => {
  let finishCreate = () => {};
  const pendingCreate = new Promise<void>((resolve) => {
    finishCreate = resolve;
  });
  const createSession = vi.fn(async () => {
    await pendingCreate;
    return HttpResponse.json({
      id: "old-session",
      created_at: "2026-09-29T10:00:00Z",
      user_id: "user-1",
      expert_id: "expert-zara",
    });
  });
  server.use(
    http.get("/api/proxy/api/experts/:expertId/onboarding", () =>
      HttpResponse.json(onboardingCard()),
    ),
    http.post("/api/proxy/api/chat/sessions", createSession),
  );
  const props = {
    target: {
      expertId: "expert-zara",
      name: "Zara",
      role: "GTM Strategist",
      avatarUrl: null,
    },
    resumeLatest: false,
    onClose: () => {},
  };
  const queryClient = new QueryClient();
  function drawer(threadKey: number) {
    return (
      <QueryClientProvider client={queryClient}>
        <ExpertChatDrawer {...props} threadKey={threadKey} />
      </QueryClientProvider>
    );
  }
  const { rerender } = render(drawer(0));
  fireEvent.click(await screen.findByRole("radio", { name: "Pricing" }));
  fireEvent.click(screen.getByRole("button", { name: "Send answers" }));
  await waitFor(() => expect(createSession).toHaveBeenCalledOnce());
  rerender(drawer(1));
  finishCreate();
  await waitFor(() => {
    expect(
      (screen.getByRole("button", { name: "Skip" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
  });
  expect(screen.getByText("Which outcome should I start with?")).toBeDefined();
});
