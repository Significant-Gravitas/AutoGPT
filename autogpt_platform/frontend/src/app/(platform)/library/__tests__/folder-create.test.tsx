import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import LibraryPage from "../page";

// The picker's virtualised emoji list renders nothing under happy-dom.
vi.mock("@/app/(platform)/library/components/LazyEmojiPicker", () => ({
  LazyEmojiPicker: ({
    onEmojiSelect,
  }: {
    onEmojiSelect: (emoji: string) => void;
  }) => (
    <button type="button" onClick={() => onEmojiSelect("📁")}>
      Pick 📁
    </button>
  ),
}));

const PAGE = { total_pages: 1, current_page: 1, page_size: 20 };
const GENERIC = "Failed to create folder. Please try again.";
const AT_THE_CAP =
  "You can have at most 1,000 folders. Delete one to make another.";

// The bodies api/utils/exceptions.py sends for a refused create.
function refusal(status: number, detail: string, hint: string) {
  return http.post("/api/proxy/api/library/folders", () =>
    HttpResponse.json(
      {
        message: "Failed to process POST /api/library/folders",
        detail,
        hint,
      },
      { status },
    ),
  );
}

test.each([
  {
    status: 400,
    detail: AT_THE_CAP,
    hint: "Adjust the request and retry.",
    shown: AT_THE_CAP,
    hidden: GENERIC,
  },
  {
    status: 500,
    detail: "Internal server error",
    hint: "Check server logs and dependent services.",
    shown: GENERIC,
    hidden: "Internal server error",
  },
])(
  "a new folder refused with $status shows $shown",
  async ({ status, detail, hint, shown, hidden }) => {
    server.use(
      getGetV2ListLibraryAgentsMockHandler({
        agents: [],
        pagination: { ...PAGE, total_items: 0 },
      }),
      getGetV2ListFavoriteLibraryAgentsMockHandler({
        agents: [],
        pagination: { ...PAGE, total_items: 0 },
      }),
      getGetV1ListAllExecutionsMockHandler([]),
      getGetV2ListLibraryFoldersMockHandler({
        folders: [],
        pagination: { ...PAGE, total_items: 0 },
      }),
      refusal(status, detail, hint),
    );
    const user = userEvent.setup();
    render(
      <>
        <LibraryPage />
        <Toaster />
      </>,
      {
        queryClient: new QueryClient({
          defaultOptions: { queries: { retry: false } },
        }),
      },
    );

    await user.click(
      await screen.findByRole("button", { name: "Create folder" }),
    );
    const dialog = await screen.findByRole("dialog", { name: "Create Folder" });
    await user.type(
      within(dialog).getByRole("textbox", { name: "Folder name" }),
      "Invoices",
    );
    fireEvent.click(within(dialog).getByRole("combobox"));
    fireEvent.click(await screen.findByRole("option", { name: "Blue" }));
    await user.click(
      await within(dialog).findByRole("button", { name: "Pick 📁" }),
    );
    const create = within(dialog).getByRole("button", { name: "Create" });
    await waitFor(() =>
      expect((create as HTMLButtonElement).disabled).toBe(false),
    );
    await user.click(create);

    expect(await screen.findByText(shown)).toBeDefined();
    expect(screen.queryByText(hidden)).toBeNull();
  },
);
