import { render, screen } from "@testing-library/react";
import { expect, test } from "vitest";
import { ExpertSessionGroup } from "./ExpertSessionGroup";

test("does not label the mixed pinned group as an AI Expert", () => {
  render(
    <ExpertSessionGroup
      groupKey="pinned"
      label="Pinned"
      sessions={[]}
      renderRow={() => null}
    />,
  );
  expect(screen.getByRole("button", { name: "Pinned" })).toBeDefined();
  expect(screen.queryByText("AI Expert")).toBeNull();
  expect(screen.queryByText("Your personal Head of AI")).toBeNull();
});
