import { expect, test, describe, vi, beforeEach, afterEach } from "vitest";
import {
  render,
  screen,
  waitFor,
  fireEvent,
  cleanup,
} from "@testing-library/react";
import { ChatSidebar } from "../ChatSidebar";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListSessionsMockHandler,
  getDeleteV2DeleteSessionMockHandler204,
  getDeleteV2DeleteSessionMockHandler422,
} from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { SidebarProvider } from "@/components/ui/sidebar";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { http, HttpResponse, delay } from "msw";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";

// Mock sessions data
const mockSessions = {
  sessions: [
    {
      id: "session-1",
      title: "First Chat",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    },
    {
      id: "session-2",
      title: "Second Chat",
      created_at: new Date(Date.now() - 86400000).toISOString(),
      updated_at: new Date(Date.now() - 86400000).toISOString(),
    },
    {
      id: "session-3",
      title: null,
      created_at: new Date(Date.now() - 172800000).toISOString(),
      updated_at: new Date(Date.now() - 172800000).toISOString(),
    },
  ],
  total: 3,
};

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
}

function TestWrapper({
  children,
  searchParams = "",
  onUrlUpdate,
}: {
  children: React.ReactNode;
  searchParams?: string;
  onUrlUpdate?: (event: { queryString: string }) => void;
}) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <BackendAPIProvider>
        <NuqsTestingAdapter
          searchParams={searchParams}
          hasMemory
          onUrlUpdate={onUrlUpdate}
        >
          <SidebarProvider defaultOpen={true}>{children}</SidebarProvider>
        </NuqsTestingAdapter>
      </BackendAPIProvider>
    </QueryClientProvider>
  );
}

function renderChatSidebar(
  searchParams = "",
  onUrlUpdate?: (event: { queryString: string }) => void,
) {
  return render(
    <TestWrapper searchParams={searchParams} onUrlUpdate={onUrlUpdate}>
      <ChatSidebar />
    </TestWrapper>,
  );
}

describe("ChatSidebar", () => {
  beforeEach(() => {
    server.use(
      getGetV2ListSessionsMockHandler(() => mockSessions),
      getDeleteV2DeleteSessionMockHandler204(),
    );
  });

  afterEach(() => {
    cleanup();
  });

  describe("Sessions List", () => {
    test("renders session list correctly", async () => {
      renderChatSidebar();

      // Use getAllByText since component may render multiple times
      await waitFor(() => {
        const elements = screen.getAllByText("First Chat");
        expect(elements.length).toBeGreaterThan(0);
      });

      expect(screen.getAllByText("Second Chat").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Untitled chat").length).toBeGreaterThan(0);
    });

    test("shows empty state when no sessions", async () => {
      server.use(
        getGetV2ListSessionsMockHandler(() => ({
          sessions: [],
          total: 0,
        })),
      );

      renderChatSidebar();

      await waitFor(() => {
        expect(
          screen.getAllByText("No conversations yet").length,
        ).toBeGreaterThan(0);
      });
    });

    test("formats dates correctly", async () => {
      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      expect(screen.getAllByText("Today").length).toBeGreaterThan(0);
      expect(screen.getAllByText("Yesterday").length).toBeGreaterThan(0);
      expect(screen.getAllByText("2 days ago").length).toBeGreaterThan(0);
    });
  });

  describe("Delete Dialog", () => {
    test("opens delete dialog when trash button is clicked", async () => {
      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      // Find and click the delete button for the first session
      const deleteButtons = screen.getAllByLabelText("Delete chat");
      fireEvent.click(deleteButtons[0]);

      // Dialog should appear with confirmation text
      await waitFor(() => {
        expect(
          screen.getByText(/Are you sure you want to delete/),
        ).toBeDefined();
      });
    });

    test("closes dialog when cancel is clicked", async () => {
      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      const deleteButtons = screen.getAllByLabelText("Delete chat");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(
          screen.getByText(/Are you sure you want to delete/),
        ).toBeDefined();
      });

      const cancelButton = screen.getByRole("button", { name: "Cancel" });
      fireEvent.click(cancelButton);

      await waitFor(() => {
        expect(
          screen.queryByText(/Are you sure you want to delete/),
        ).toBeNull();
      });
    });

    test("calls delete API when delete button is clicked", async () => {
      const deleteMock = vi.fn();
      server.use(
        http.delete(
          "http://localhost:3000/api/proxy/api/chat/sessions/:sessionId",
          async ({ params }) => {
            deleteMock(params.sessionId);
            return new HttpResponse(null, { status: 204 });
          },
        ),
      );

      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      const deleteButtons = screen.getAllByLabelText("Delete chat");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(
          screen.getByText(/Are you sure you want to delete/),
        ).toBeDefined();
      });

      const deleteButton = screen.getByRole("button", { name: "Delete" });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(deleteMock).toHaveBeenCalledWith("session-1");
      });
    });

    test("closes dialog after successful deletion", async () => {
      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      const deleteButtons = screen.getAllByLabelText("Delete chat");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(
          screen.getByText(/Are you sure you want to delete/),
        ).toBeDefined();
      });

      const deleteButton = screen.getByRole("button", { name: "Delete" });
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(
          screen.queryByText(/Are you sure you want to delete/),
        ).toBeNull();
      });
    });

    test("handles deletion error gracefully", async () => {
      server.use(getDeleteV2DeleteSessionMockHandler422());

      renderChatSidebar();

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      const deleteButtons = screen.getAllByLabelText("Delete chat");
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(
          screen.getByText(/Are you sure you want to delete/),
        ).toBeDefined();
      });

      const deleteButton = screen.getByRole("button", { name: "Delete" });
      fireEvent.click(deleteButton);

      // Dialog should close even on error
      await waitFor(() => {
        expect(
          screen.queryByText(/Are you sure you want to delete/),
        ).toBeNull();
      });
    });
  });

  describe("Session Selection", () => {
    test("highlights currently selected session", async () => {
      renderChatSidebar("?sessionId=session-1");

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      // Find the session with selected styling (text-zinc-600 indicates selected)
      const selectedSessions = screen.getAllByText("First Chat");
      const hasSelectedStyle = selectedSessions.some(
        (el) =>
          el.className.includes("text-zinc-600") ||
          el.closest("div")?.className.includes("bg-zinc-100"),
      );
      expect(hasSelectedStyle).toBe(true);
    });

    test("selects session when clicked", async () => {
      const urlUpdateSpy = vi.fn();

      renderChatSidebar("", urlUpdateSpy);

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      // Find a session button and click it
      const sessionElements = screen.getAllByText("First Chat");
      const sessionButton = sessionElements[0].closest("button");
      if (sessionButton) {
        fireEvent.click(sessionButton);
      }

      await waitFor(() => {
        expect(urlUpdateSpy).toHaveBeenCalled();
        const lastCall =
          urlUpdateSpy.mock.calls[urlUpdateSpy.mock.calls.length - 1];
        expect(lastCall[0].queryString).toContain("sessionId=session-1");
      });
    });
  });

  describe("New Chat", () => {
    test("clears session selection when new chat button is clicked", async () => {
      const urlUpdateSpy = vi.fn();

      renderChatSidebar("?sessionId=session-1", urlUpdateSpy);

      await waitFor(() => {
        expect(screen.getAllByText("First Chat").length).toBeGreaterThan(0);
      });

      const newChatButton = screen.getByRole("button", { name: "New Chat" });
      fireEvent.click(newChatButton);

      await waitFor(() => {
        expect(urlUpdateSpy).toHaveBeenCalled();
        const lastCall =
          urlUpdateSpy.mock.calls[urlUpdateSpy.mock.calls.length - 1];
        expect(lastCall[0].queryString).not.toContain("sessionId=session-1");
      });
    });
  });
});
