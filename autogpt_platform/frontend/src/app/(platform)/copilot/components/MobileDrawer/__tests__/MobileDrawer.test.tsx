import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import {
  afterAll,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { useCopilotUIStore } from "../../../store";
import { MobileDrawer } from "../MobileDrawer";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "chat-search",
  };
});

const sessions = [
  {
    id: "s1",
    title: "Budget notes",
    is_processing: false,
    source_platform: "slack",
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
  },
  {
    id: "s2",
    title: "Revenue forecast",
    is_processing: false,
    created_at: "2025-01-02T00:00:00Z",
    updated_at: "2025-01-02T00:00:00Z",
  },
];

describe("MobileDrawer search", () => {
  const originalSetPointerCapture = HTMLElement.prototype.setPointerCapture;
  const originalReleasePointerCapture =
    HTMLElement.prototype.releasePointerCapture;
  const originalHasPointerCapture = HTMLElement.prototype.hasPointerCapture;

  beforeAll(() => {
    HTMLElement.prototype.setPointerCapture = vi.fn();
    HTMLElement.prototype.releasePointerCapture = vi.fn();
    HTMLElement.prototype.hasPointerCapture = vi.fn(() => false);
  });

  afterAll(() => {
    HTMLElement.prototype.setPointerCapture = originalSetPointerCapture;
    HTMLElement.prototype.releasePointerCapture = originalReleasePointerCapture;
    HTMLElement.prototype.hasPointerCapture = originalHasPointerCapture;
  });

  beforeEach(() => {
    useCopilotUIStore.setState({
      isDrawerOpen: true,
      isSearchOpen: false,
      completedSessionIDs: new Set<string>(),
    });
    server.use(
      getGetV2ListSessionsMockHandler200({
        sessions,
        total: sessions.length,
      }),
    );
  });

  it("opens and closes the inline search view", async () => {
    render(<MobileDrawer />);

    expect(await screen.findByText("Budget notes")).toBeDefined();
    expect(screen.getByAltText("Slack")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /search chats/i }));
    expect(
      screen.getByRole("textbox", { name: /search chats/i }),
    ).toBeDefined();

    fireEvent.change(screen.getByRole("textbox", { name: /search chats/i }), {
      target: { value: "forecast" },
    });
    expect(
      await screen.findByRole("option", { name: /revenue forecast/i }),
    ).toBeDefined();
    await vi.waitFor(() => {
      expect(
        screen.queryByRole("option", { name: /budget notes/i }),
      ).toBeNull();
    });

    fireEvent.click(screen.getByRole("button", { name: /close search/i }));
    expect(screen.queryByRole("textbox", { name: /search chats/i })).toBeNull();
    expect(await screen.findByText("Budget notes")).toBeDefined();
  });

  it("selects a filtered session and closes the drawer", async () => {
    render(<MobileDrawer />);

    fireEvent.click(
      await screen.findByRole("button", { name: /search chats/i }),
    );
    fireEvent.change(screen.getByRole("textbox", { name: /search chats/i }), {
      target: { value: "revenue" },
    });
    fireEvent.click(
      await screen.findByRole("option", { name: /revenue forecast/i }),
    );

    await vi.waitFor(() => {
      expect(useCopilotUIStore.getState().isDrawerOpen).toBe(false);
      expect(useCopilotUIStore.getState().isSearchOpen).toBe(false);
    });
  });
});
