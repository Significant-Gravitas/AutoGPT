import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SkillLoadedCard } from "../SkillLoadedCard";

describe("SkillLoadedCard", () => {
  afterEach(cleanup);

  it("names the exact loaded version and origin and links to its change", () => {
    render(
      <SkillLoadedCard
        output={{
          name: "csv-import-checks",
          version: 3,
          version_id: "ver-3",
          origin: "saved_overnight",
          origin_label: "Saved overnight",
          expert_id: "expert-1",
        }}
      />,
    );

    expect(
      screen.getByText("Skill loaded: csv-import-checks v3 · Saved overnight"),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Loaded for this task; loading is not a success record.",
      ),
    ).toBeDefined();
    const link = screen.getByRole("link", { name: "View change" });
    expect(link.getAttribute("href")).toBe(
      "/team/expert-1?tab=skills&skill=csv-import-checks&version=ver-3",
    );
  });

  it("renders nothing for an unversioned load", () => {
    const { container } = render(
      <SkillLoadedCard output={{ name: "agent_building_guide" }} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("links personal skills to the memory page's learning detail", () => {
    render(
      <SkillLoadedCard
        output={{
          name: "deploy-notes",
          version: 1,
          origin_label: "Saved during work",
        }}
      />,
    );
    expect(
      screen.getByRole("link", { name: "View change" }).getAttribute("href"),
    ).toBe("/settings/memory?skill=deploy-notes");
  });
});
