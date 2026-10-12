import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  act,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, test, vi } from "vitest";
import { ChatContainer, type ChatContainerProps } from "../ChatContainer";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const props: ChatContainerProps = {
  messages: [],
  status: "ready",
  error: undefined,
  sessionId: "empty-session",
  isLoadingSession: false,
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  onStop: vi.fn(),
  onSend: vi.fn(),
  expertIdentity: {
    id: "expert-kepler",
    name: "Kepler",
    role: "researcher",
    avatarUrl: null,
    isArchived: false,
    readOnlyReason: null,
  },
};

function mockOnboarding(completed = false) {
  const request = vi.fn(() =>
    HttpResponse.json(
      completed
        ? null
        : {
            type: "expert_onboarding",
            expert_id: "expert-kepler",
            greeting: "I'm Kepler.",
            message: "What topic?",
            steps: [
              {
                question: "What topic?",
                keyword: "topic",
                options: ["AI", "Energy"],
              },
            ],
          },
    ),
  );
  server.use(http.get("/api/proxy/api/experts/:expertId/onboarding", request));
  return request;
}

describe("new full-page expert chats", () => {
  test("keeps draft answers when onboarding status refetches", async () => {
    mockOnboarding();
    const queryClient = new QueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <ChatContainer {...props} />
      </QueryClientProvider>,
    );
    fireEvent.click(await screen.findByRole("radio", { name: "AI" }));
    await act(async () => {
      await queryClient.invalidateQueries({
        queryKey: ["/api/experts/expert-kepler/onboarding"],
      });
    });
    expect(
      screen.getByRole("radio", { name: "AI" }).getAttribute("aria-checked"),
    ).toBe("true");
  });

  test("shows pending onboarding in an empty chat", async () => {
    mockOnboarding();
    const onSend = vi.fn();
    render(<ChatContainer {...props} onSend={onSend} />);
    expect(await screen.findByText("What topic?")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Skip" }));
    await waitFor(() =>
      expect(onSend).toHaveBeenCalledWith(
        "Let's skip the setup questions for now.",
      ),
    );
  });

  test("keeps the home hero when an expert is picked before a chat exists", async () => {
    const request = mockOnboarding();
    render(<ChatContainer {...props} sessionId={null} />);
    expect(await screen.findByRole("textbox")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
    expect(screen.queryByText("What topic?")).toBeNull();
  });

  test("sends completed answers through the new chat's send handler", async () => {
    mockOnboarding();
    const onSend = vi.fn();
    render(<ChatContainer {...props} onSend={onSend} />);
    fireEvent.click(await screen.findByRole("radio", { name: "AI" }));
    fireEvent.click(screen.getByRole("button", { name: /Send answers/i }));
    await waitFor(() =>
      expect(onSend).toHaveBeenCalledWith(
        "**Here are my answers:**\n\n> What topic?\n\nAI\n\nPlease proceed.",
      ),
    );
  });

  test("does not show completed or skipped onboarding", async () => {
    const request = mockOnboarding(true);
    render(<ChatContainer {...props} />);
    await waitFor(() => expect(request).toHaveBeenCalled());
    expect(screen.queryByText("What topic?")).toBeNull();
  });

  test("does not request onboarding after the user sends a prompt", async () => {
    const request = mockOnboarding();
    render(
      <ChatContainer
        {...props}
        sessionId="existing-session"
        messages={[
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "Research this company" }],
          },
        ]}
      />,
    );
    expect(await screen.findByText("Research this company")).toBeDefined();
    expect(request).not.toHaveBeenCalled();
    expect(screen.queryByText("What topic?")).toBeNull();
  });
});
