import { HttpResponse, http } from "msw";
import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getGetV2ListLibraryAgentsResponseMock } from "@/app/api/__generated__/endpoints/library/library.msw";
import LibraryPage from "../page";

describe("LibraryPage while its list loads", () => {
  test("shows skeleton cards in the list's place, then the workflows", async () => {
    const mock = getGetV2ListLibraryAgentsResponseMock();
    const list = {
      ...mock,
      agents: [{ ...mock.agents[0], name: "Harbour tide alerts" }],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    };
    // The page and the list each ask for it; both wait until the test answers.
    const held: Array<() => void> = [];
    server.use(
      http.get(
        "/api/proxy/api/library/agents",
        () =>
          new Promise<Response>((resolve) => {
            held.push(() => resolve(HttpResponse.json(list)));
          }),
      ),
    );

    const { container } = render(<LibraryPage />);

    const status = await screen.findByRole("status", {
      name: "Loading workflows…",
    });
    expect(
      status.querySelectorAll('[data-testid="library-agent-card-skeleton"]'),
    ).toHaveLength(6);
    expect(container.querySelector('[data-slot="loading-spinner"]')).toBeNull();

    held.forEach((answer) => answer());

    expect(await screen.findByText("Harbour tide alerts")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByTestId("library-agent-list-skeleton")).toBeNull(),
    );
  });
});
