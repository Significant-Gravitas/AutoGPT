import { render } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import HomeLoading from "../(home)/loading";
import AgentLoading from "../agent/loading";
import CreatorLoading from "../creator/loading";
import SearchLoading from "../search/loading";
import SkillsLoading from "../skills/loading";

// The expert page deliberately has no loading state: a loading boundary
// streams its content in a hidden chunk that a crawler without JavaScript
// never sees. Every other marketplace route keeps the shared skeleton.
describe("Marketplace loading states", () => {
  test.each([
    ["home", HomeLoading],
    ["agent", AgentLoading],
    ["creator", CreatorLoading],
    ["search", SearchLoading],
    ["skills", SkillsLoading],
  ])("%s shows the marketplace skeleton", (_route, Loading) => {
    const { container } = render(<Loading />);

    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(
      0,
    );
  });
});
