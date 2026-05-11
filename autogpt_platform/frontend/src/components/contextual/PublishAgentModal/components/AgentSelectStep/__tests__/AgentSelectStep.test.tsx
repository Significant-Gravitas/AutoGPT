import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isLoggedIn: true }),
}));

import { AgentSelectStep } from "../AgentSelectStep";

function fakeAgents(total: number, page: number, pageSize: number) {
  const start = (page - 1) * pageSize;
  return Array.from({ length: Math.min(pageSize, total - start) }, (_, i) => {
    const idx = start + i + 1;
    return {
      graph_id: `graph-${idx}`,
      graph_version: 1,
      agent_name: `Agent ${idx}`,
      description: `Description ${idx}`,
      last_edited: new Date(
        `2026-05-${String(((idx - 1) % 28) + 1).padStart(2, "0")}T00:00:00Z`,
      ).toISOString(),
      agent_image: null,
      recommended_schedule_cron: null,
    };
  });
}

function installHandler(total = 47, pageSize = 10) {
  server.use(
    http.get(
      "http://localhost:3000/api/proxy/api/store/my-unpublished-agents",
      ({ request }) => {
        const url = new URL(request.url);
        const page = Number(url.searchParams.get("page") ?? "1");
        const size = Number(url.searchParams.get("page_size") ?? pageSize);
        return HttpResponse.json({
          agents: fakeAgents(total, page, size),
          pagination: {
            current_page: page,
            total_items: total,
            total_pages: Math.ceil(total / size),
            page_size: size,
          },
        });
      },
    ),
  );
}

describe("AgentSelectStep", () => {
  const onSelect = vi.fn();
  const onCancel = vi.fn();
  const onNext = vi.fn();
  const onOpenBuilder = vi.fn();

  beforeEach(() => {
    onSelect.mockReset();
    onCancel.mockReset();
    onNext.mockReset();
    onOpenBuilder.mockReset();
  });

  it("renders the loaded agents and a pagination bar when totalPages > 1", async () => {
    installHandler(47, 10);
    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    expect(
      await screen.findByText("Agent 1", {}, { timeout: 3000 }),
    ).toBeDefined();
    expect(screen.getByText("Agent 10")).toBeDefined();
    expect(screen.getByText("1–10 of 47")).toBeDefined();
    expect(screen.getByRole("button", { name: "Next page" })).toBeDefined();
  });

  it("selecting an agent enables Continue and forwards data via onNext", async () => {
    installHandler(12, 10);
    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    const card = await screen.findByText("Agent 1", {}, { timeout: 3000 });
    fireEvent.click(card);
    expect(onSelect).toHaveBeenCalledWith("graph-1", 1);

    const continueBtn = screen.getByRole("button", { name: "Continue" });
    await waitFor(() =>
      expect(continueBtn).not.toHaveProperty("disabled", true),
    );

    fireEvent.click(continueBtn);
    expect(onNext).toHaveBeenCalledWith(
      "graph-1",
      1,
      expect.objectContaining({ name: "Agent 1" }),
    );
  });

  it("clicking the Next page button advances to page 2", async () => {
    installHandler(47, 10);
    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    await screen.findByText("Agent 1", {}, { timeout: 3000 });

    fireEvent.click(screen.getByRole("button", { name: "Next page" }));

    await waitFor(
      () => {
        expect(screen.queryByText("11–20 of 47")).not.toBeNull();
      },
      { timeout: 3000 },
    );
  });

  it("shows the empty state when the user has no agents", async () => {
    server.use(
      http.get(
        "http://localhost:3000/api/proxy/api/store/my-unpublished-agents",
        () =>
          HttpResponse.json({
            agents: [],
            pagination: {
              current_page: 1,
              total_items: 0,
              total_pages: 0,
              page_size: 10,
            },
          }),
      ),
    );

    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    expect(
      await screen.findByText(
        "No publishable agents yet",
        {},
        { timeout: 3000 },
      ),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Open builder" }));
    expect(onOpenBuilder).toHaveBeenCalled();
  });

  it("renders the ellipsis page range when total pages exceeds 7", async () => {
    // 100 items / 10 per page = 10 pages → ellipsis branch is exercised.
    installHandler(100, 10);
    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    await screen.findByText("Agent 1", {}, { timeout: 3000 });

    // The ellipsis character should show up between page chunks.
    expect(screen.getAllByText("…").length).toBeGreaterThan(0);

    // First and last pages are always shown.
    expect(
      screen.getByRole("button", { name: "1", current: "page" }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "10" })).toBeDefined();

    // Clicking a numeric page button jumps to that page.
    fireEvent.click(screen.getByRole("button", { name: "10" }));
    await waitFor(
      () => {
        expect(screen.queryByText("91–100 of 100")).not.toBeNull();
      },
      { timeout: 3000 },
    );
  });

  it("renders the error state when the API fails", async () => {
    server.use(
      http.get(
        "http://localhost:3000/api/proxy/api/store/my-unpublished-agents",
        () => HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(
      <AgentSelectStep
        onSelect={onSelect}
        onCancel={onCancel}
        onNext={onNext}
        onOpenBuilder={onOpenBuilder}
      />,
    );

    expect(
      await screen.findByText(
        "We could not load your agents",
        {},
        { timeout: 3000 },
      ),
    ).toBeDefined();
  });
});
