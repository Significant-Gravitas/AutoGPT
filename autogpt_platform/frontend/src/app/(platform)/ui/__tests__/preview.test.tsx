import { render, screen } from "@/tests/integrations/test-utils";
import { expect, test } from "vitest";
import { ToolUIPreview } from "../components/ToolUIPreview/ToolUIPreview";

test("previews every expert tool card and the question item", async () => {
  render(<ToolUIPreview />);

  expect(screen.getByText("hire_expert · preview")).toBeDefined();
  expect(screen.getByText("Ready to hire")).toBeDefined();
  expect(screen.getAllByText("Scout").length).toBeGreaterThan(0);

  expect(screen.getByText('Ready to raise "Otto"')).toBeDefined();
  expect(screen.getByText("Added to the team")).toBeDefined();
  expect(
    screen.getByText('Handed over: "Take over the weekly inbox summary"'),
  ).toBeDefined();

  expect(screen.getByText("Scout has a question")).toBeDefined();
});
