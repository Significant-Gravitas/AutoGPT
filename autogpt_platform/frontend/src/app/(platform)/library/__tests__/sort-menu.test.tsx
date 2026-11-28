import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { expect, test, vi } from "vitest";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import LibraryPage from "../page";

const PAGE = { total_pages: 1, current_page: 1, page_size: 20 };

// The library, recording the sort each list request asks for.
function serveLibrary() {
  const sorts: (string | null)[] = [];
  const agent = {
    ...getGetV2ListLibraryAgentsResponseMock().agents[0],
    name: "Weekly report",
  };
  server.use(
    getGetV2ListLibraryAgentsMockHandler(({ request }) => {
      sorts.push(new URL(request.url).searchParams.get("sort_by"));
      return { agents: [agent], pagination: { ...PAGE, total_items: 1 } };
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      agents: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV2ListLibraryFoldersMockHandler({
      folders: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
  return sorts;
}

function sortMenu() {
  return within(screen.getByTestId("sort-by-dropdown")).getByRole("combobox");
}

test.each([
  ["?sort=createdAt", "createdAt", "Creation Date"],
  ["?sort=lastRunAt", "lastRunAt", "Last Run"],
])(
  "a Library opened at %s names that sort in the sort menu",
  async (searchParams, sortBy, label) => {
    const sorts = serveLibrary();

    render(
      <NuqsTestingAdapter searchParams={searchParams}>
        <LibraryPage />
      </NuqsTestingAdapter>,
    );

    await screen.findByText("Weekly report");
    expect(sorts).toContain(sortBy);
    expect(sortMenu().textContent).toBe(label);
  },
);

test("a Library opened without a sort names its default, Last Modified", async () => {
  const sorts = serveLibrary();

  render(<LibraryPage />);

  await screen.findByText("Weekly report");
  expect(sorts).toContain("updatedAt");
  expect(sortMenu().textContent).toBe("Last Modified");
});

test("a Library opened at an unknown sort lists and names Last Modified", async () => {
  const sorts = serveLibrary();

  render(
    <NuqsTestingAdapter searchParams="?sort=bogus">
      <LibraryPage />
    </NuqsTestingAdapter>,
  );

  await screen.findByText("Weekly report");
  expect(sorts).toContain("updatedAt");
  expect(sorts).not.toContain("bogus");
  expect(sortMenu().textContent).toBe("Last Modified");
});

test("a sort picked in the menu goes into the URL and stays named", async () => {
  const sorts = serveLibrary();
  const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
  const user = userEvent.setup();

  render(
    <NuqsTestingAdapter
      searchParams="?sort=updatedAt"
      hasMemory
      onUrlUpdate={onUrlUpdate}
    >
      <LibraryPage />
    </NuqsTestingAdapter>,
  );

  await screen.findByText("Weekly report");
  await user.click(sortMenu());
  await user.click(await screen.findByRole("option", { name: "Last Run" }));

  await waitFor(() => expect(sorts).toContain("lastRunAt"));
  await waitFor(() =>
    expect(
      onUrlUpdate.mock.calls.some(
        ([event]) => event.searchParams.get("sort") === "lastRunAt",
      ),
    ).toBe(true),
  );
  expect(sortMenu().textContent).toBe("Last Run");
});
