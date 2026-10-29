import { describe, expect, test } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { LibraryAgentCard } from "../LibraryAgentCard";
import { FavoriteAnimationProvider } from "../../../context/FavoriteAnimationContext";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { AgentStatusInfo } from "../../../types";

const AGENT = {
  id: "4f6b2c1e-0000-4000-8000-000000000001",
  name: "Daily digest",
  image_url: "https://cdn.test/agent.png",
  is_favorite: false,
} as unknown as LibraryAgent;

const STATUS: AgentStatusInfo = {
  status: "idle",
  health: "good",
  progress: null,
  totalRuns: 3,
  lastRunAt: null,
  lastError: null,
  activeExecutionID: null,
  monthlySpend: 0,
  nextScheduledRun: null,
  triggerType: null,
};

function renderCard() {
  return render(
    <FavoriteAnimationProvider>
      <LibraryAgentCard agent={AGENT} statusInfo={STATUS} />
    </FavoriteAnimationProvider>,
  );
}

describe("LibraryAgentCard", () => {
  test("renders the agent image while the URL loads", () => {
    renderCard();
    expect(screen.getByAltText("Daily digest preview image")).toBeDefined();
  });

  test("falls back to the gradient tile when the image URL is dead", () => {
    renderCard();

    fireEvent.error(screen.getByAltText("Daily digest preview image"));

    expect(screen.queryByAltText("Daily digest preview image")).toBeNull();
    expect(screen.getByText("Daily digest")).toBeDefined();
  });
});
