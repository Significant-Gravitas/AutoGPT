import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { render, screen, fireEvent } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListFavoriteLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV1GetUserCreditsMockHandler,
  getGetV1GetAutoTopUpMockHandler,
} from "@/app/api/__generated__/endpoints/credits/credits.msw";
import { Key } from "@/services/storage/local-storage";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import LibraryPage from "../page";

// Billing must be on for the provider to derive `isOutOfCredits`; keep the real
// `Flag` enum so other flags the page reads (e.g. AGENT_BRIEFING) resolve.
vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => flag === actual.Flag.ENABLE_PLATFORM_PAYMENT,
  };
});

function setupLibraryHandlers() {
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      ...getGetV2ListFavoriteLibraryAgentsResponseMock(),
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 10,
      },
    }),
    getGetV2ListLibraryFoldersMockHandler({
      folders: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
}

function setupCredits(credits: number) {
  server.use(
    getGetV1GetUserCreditsMockHandler({ credits }),
    getGetV1GetAutoTopUpMockHandler({ amount: 0, threshold: 0 }),
  );
}

function renderLibraryWithTopUpPrompt() {
  return render(
    <TopUpPromptProvider>
      <LibraryPage />
    </TopUpPromptProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  setupLibraryHandlers();
});

afterEach(() => {
  localStorage.clear();
});

describe("Library low-credit banner", () => {
  test("shows the banner when the user is out of credits", async () => {
    // Suppress the daily auto-opener so only the banner is under test.
    localStorage.setItem(
      Key.TOP_UP_MODAL_LAST_SHOWN,
      new Date().toDateString(),
    );
    setupCredits(0);

    renderLibraryWithTopUpPrompt();

    expect(await screen.findByText(/out of automation credits/i)).toBeDefined();
  });

  test("the Top up CTA opens the top-up dialog", async () => {
    localStorage.setItem(
      Key.TOP_UP_MODAL_LAST_SHOWN,
      new Date().toDateString(),
    );
    setupCredits(0);

    renderLibraryWithTopUpPrompt();

    await screen.findByText(/out of automation credits/i);

    fireEvent.click(screen.getByRole("button", { name: /top up/i }));

    // The dialog body copy mentions Autopilot, which the banner copy does not —
    // keeps this assertion unambiguous against the banner's own message.
    expect(
      await screen.findByText(/keep your agents and Autopilot/i),
    ).toBeDefined();
  });

  test("hides the banner when the user still has credits", async () => {
    setupCredits(500);

    renderLibraryWithTopUpPrompt();

    // Let the header render so the page has fully mounted before asserting absence.
    await screen.findAllByTestId("import-button");

    expect(screen.queryByText(/out of automation credits/i)).toBeNull();
  });
});
