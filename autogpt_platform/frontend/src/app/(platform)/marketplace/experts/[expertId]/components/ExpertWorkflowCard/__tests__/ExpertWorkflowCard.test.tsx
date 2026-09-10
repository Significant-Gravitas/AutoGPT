import { describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { ExpertWorkflowCard } from "../ExpertWorkflowCard";
import type { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";

vi.mock("../useExpertWorkflowCard", () => ({
  useExpertWorkflowCard: () => ({
    imageUrl: "https://cdn.test/workflow.png",
    isLoadingImage: false,
  }),
}));

const WORKFLOW = {
  name: "Daily digest",
  description: "Sends a summary each morning",
  store_listing_version_id: "lv1",
} as ExpertWorkflowRef;

describe("ExpertWorkflowCard", () => {
  test("falls back to the accent icon when the image URL is dead", () => {
    const { container } = render(
      <ExpertWorkflowCard
        workflow={WORKFLOW}
        accent={getExpertAccent("Ops")}
      />,
    );

    expect(screen.getByAltText("Daily digest preview image")).toBeDefined();
    fireEvent.error(screen.getByAltText("Daily digest preview image"));

    expect(screen.queryByAltText("Daily digest preview image")).toBeNull();
    expect(container.querySelector("svg")).not.toBeNull();
    expect(screen.getByText("Daily digest")).toBeDefined();
  });
});
