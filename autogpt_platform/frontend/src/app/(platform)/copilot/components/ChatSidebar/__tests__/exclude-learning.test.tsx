import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { getExcludeSkillLearningSourceMockHandler200 } from "@/app/api/__generated__/endpoints/skill-learning/skill-learning.msw";
import { SidebarProvider } from "@/components/ui/sidebar";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ChatSidebar } from "../ChatSidebar";

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return { ...actual, toast: (...args: unknown[]) => toastMock(...args) };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === "dream-skill-learning-enabled",
  };
});

vi.mock("../../UsageLimits/UsageLimits", () => ({ UsageLimits: () => null }));
vi.mock("../../UsageLimits/UsagePopover/UsagePopover", () => ({
  UsagePopover: () => null,
}));
vi.mock("../components/NotificationToggle/NotificationToggle", () => ({
  NotificationToggle: () => null,
}));

const sessions = [
  {
    id: "s1",
    title: "CSV import chat",
    is_processing: false,
    created_at: "2026-09-13T00:00:00Z",
    updated_at: "2026-09-13T00:00:00Z",
  },
];

describe("ChatSidebar — exclude from learning", () => {
  beforeEach(() => {
    toastMock.mockClear();
    useCopilotUIStore.setState({ sessionToDelete: null, isSearchOpen: false });
    server.use(
      getGetV2ListSessionsMockHandler200({ sessions, total: sessions.length }),
    );
  });

  afterEach(() => {
    server.resetHandlers();
  });

  it("excludes the chosen chat and confirms with a toast", async () => {
    const exclude = vi.fn();
    server.use(
      getExcludeSkillLearningSourceMockHandler200(({ params }) => {
        exclude(params);
        return {
          source_id: "src-1",
          excluded: true,
          invalidated_version_ids: [],
        };
      }),
    );
    render(
      <SidebarProvider>
        <ChatSidebar />
      </SidebarProvider>,
    );

    const row = (await screen.findByText("CSV import chat")).closest(
      "div.group",
    );
    if (!row) throw new Error("row not found");
    fireEvent.pointerDown(
      within(row as HTMLElement).getByRole("button", { name: /more actions/i }),
      { button: 0 },
    );
    fireEvent.click(
      await screen.findByRole("menuitem", { name: /exclude from learning/i }),
    );

    await waitFor(() => expect(exclude).toHaveBeenCalledTimes(1));
    expect(exclude.mock.calls[0][0]).toMatchObject({
      sourceKind: "chat_session",
      sourceRef: "s1",
    });
    await waitFor(() =>
      expect(toastMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Excluded from learning" }),
      ),
    );
  });
});
