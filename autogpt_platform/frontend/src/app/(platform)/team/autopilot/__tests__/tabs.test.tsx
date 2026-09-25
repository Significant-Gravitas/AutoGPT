import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV2ListLibraryAgentsMockHandler200 } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, UrlUpdateEvent } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, test, vi } from "vitest";
import AutopilotPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useFlagStatus: () => ({ enabled: true, ready: true }),
}));

beforeEach(() => {
  server.use(
    getListExpertsMockHandler([]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getGetV2ListLibraryAgentsMockHandler200({
      agents: [],
      pagination: {
        total_items: 0,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    }),
    getListCopilotSkillsMockHandler200([]),
  );
});

describe("Otto tab navigation", () => {
  test("opens Workflows directly from the migration notice's destination", async () => {
    render(
      <NuqsTestingAdapter searchParams="?tab=workflows">
        <AutopilotPage />
      </NuqsTestingAdapter>,
    );

    expect(await screen.findByText("Otto's Workflows")).toBeDefined();
    expect(
      screen
        .getByRole("tab", { name: "Workflows" })
        .getAttribute("aria-selected"),
    ).toBe("true");
    expect(screen.queryByText("Identity")).toBeNull();
  });

  test.each(["", "?tab=unknown"])(
    "defaults to Basics for %s",
    async (searchParams) => {
      render(
        <NuqsTestingAdapter searchParams={searchParams}>
          <AutopilotPage />
        </NuqsTestingAdapter>,
      );

      expect(await screen.findByText("Identity")).toBeDefined();
      expect(
        screen
          .getByRole("tab", { name: "Basics" })
          .getAttribute("aria-selected"),
      ).toBe("true");
    },
  );

  test("follows query changes after the page has mounted", async () => {
    const { rerender } = render(
      <NuqsTestingAdapter searchParams="?tab=workflows" hasMemory>
        <AutopilotPage />
      </NuqsTestingAdapter>,
    );
    await screen.findByText("Otto's Workflows");

    rerender(
      <NuqsTestingAdapter searchParams="?tab=basics" hasMemory>
        <AutopilotPage />
      </NuqsTestingAdapter>,
    );

    expect(await screen.findByText("Identity")).toBeDefined();
    expect(screen.queryByText("Otto's Workflows")).toBeNull();
  });

  test("writes tab navigation to history and preserves other query parameters", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();
    render(
      <NuqsTestingAdapter
        searchParams="?tab=workflows&source=migration"
        onUrlUpdate={onUrlUpdate}
        hasMemory
      >
        <AutopilotPage />
      </NuqsTestingAdapter>,
    );

    await user.click(await screen.findByRole("tab", { name: "Schedules" }));

    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    const update = onUrlUpdate.mock.calls.at(-1)?.[0];
    expect(update?.searchParams.get("tab")).toBe("schedules");
    expect(update?.searchParams.get("source")).toBe("migration");
    expect(update?.options.history).toBe("push");

    await user.click(screen.getByRole("tab", { name: "Basics" }));

    await waitFor(() =>
      expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.has("tab")).toBe(
        false,
      ),
    );
    expect(screen.getByText("Identity")).toBeDefined();
  });
});
