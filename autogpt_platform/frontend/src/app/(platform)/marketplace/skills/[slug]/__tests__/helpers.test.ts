import { describe, expect, test } from "vitest";
import { prepareSkillBody } from "../components/helpers";

describe("prepareSkillBody", () => {
  test("drops a leading title that repeats the page's", () => {
    const body = "# Outreach playbook\n\nFour sentences, no more.";

    expect(prepareSkillBody(body, "Outreach playbook")).toBe(
      "Four sentences, no more.",
    );
  });

  test("keeps a leading heading that says something else", () => {
    const body = "# Before you start\n\nRead this.";

    expect(prepareSkillBody(body, "Outreach playbook")).toContain(
      "### Before you start",
    );
  });

  test("lands the shallowest heading on h3 whatever level it starts at", () => {
    // A "##"-only body must not start at h4 just because it skipped h1.
    const shifted = prepareSkillBody("## Research\n\n### Notes", "Anything");

    expect(shifted).toContain("### Research");
    expect(shifted).toContain("#### Notes");
  });

  test("keeps relative depth when the body starts at h1", () => {
    const shifted = prepareSkillBody("# Top\n\n## Under", "Anything");

    expect(shifted).toContain("### Top");
    expect(shifted).toContain("#### Under");
  });

  test("leaves hashes inside a fenced block alone", () => {
    const body = "## Real\n\n```bash\n# not a heading\n```";
    const shifted = prepareSkillBody(body, "Anything");

    expect(shifted).toContain("### Real");
    expect(shifted).toContain("# not a heading");
  });

  test("returns a body with no headings unchanged", () => {
    expect(prepareSkillBody("Just prose.", "Anything")).toBe("Just prose.");
  });
});
