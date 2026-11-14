import { getExcludeSkillLearningSourceMockHandler200 } from "@/app/api/__generated__/endpoints/skill-learning/skill-learning.msw";
import { SidebarProvider } from "@/components/molecules/Sidebar/Sidebar";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RecentChats } from "../RecentChats";
import { sidebarBackend } from "./sidebarBackend";

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

const chats = [{ id: "s1", title: "CSV import chat" }];

describe("RecentChats — exclude from learning", () => {
  beforeEach(() => {
    toastMock.mockClear();
    server.use(...sidebarBackend(() => chats));
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
        <RecentChats />
      </SidebarProvider>,
    );

    const row = (await screen.findByText("CSV import chat")).closest("li");
    if (!row) throw new Error("row not found");
    fireEvent.pointerDown(
      within(row as HTMLElement).getByRole("button", { name: /chat actions/i }),
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
