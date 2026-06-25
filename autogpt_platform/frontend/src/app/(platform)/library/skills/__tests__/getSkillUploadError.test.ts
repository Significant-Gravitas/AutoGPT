import { describe, expect, test } from "vitest";
import {
  getSkillUploadError,
  MAX_SKILL_DESCRIPTION_CHARS,
} from "../components/UploadSkillButton/helpers";

function md(frontmatter: string, body = "# Body\n\ncontent"): string {
  return `---\n${frontmatter}\n---\n\n${body}`;
}

describe("getSkillUploadError", () => {
  test("returns null for a valid SKILL.md", () => {
    expect(
      getSkillUploadError(md("name: ok_skill\ndescription: A short hook.")),
    ).toBeNull();
  });

  test("flags a file with no frontmatter", () => {
    const err = getSkillUploadError("just prose, no frontmatter block");
    expect(err).toContain("not a valid SKILL.md");
  });

  test("reports the exact length when description is too long", () => {
    const desc = "x".repeat(MAX_SKILL_DESCRIPTION_CHARS + 1);
    const err = getSkillUploadError(md(`name: ok_skill\ndescription: ${desc}`));
    expect(err).toContain(`${MAX_SKILL_DESCRIPTION_CHARS + 1}/200`);
    expect(err).toContain("trim at least 1");
  });

  test("accepts a description exactly at the limit", () => {
    const desc = "x".repeat(MAX_SKILL_DESCRIPTION_CHARS);
    expect(
      getSkillUploadError(md(`name: ok_skill\ndescription: ${desc}`)),
    ).toBeNull();
  });

  test("strips surrounding quotes before measuring", () => {
    const desc = "x".repeat(MAX_SKILL_DESCRIPTION_CHARS);
    expect(
      getSkillUploadError(md(`name: ok_skill\ndescription: "${desc}"`)),
    ).toBeNull();
  });

  test("does not false-reject a block-scalar description it cannot measure", () => {
    const block = `name: ok_skill\ndescription: |\n  ${"x".repeat(300)}`;
    // Block scalar is unmeasurable from one line — defer to the backend.
    expect(getSkillUploadError(md(block))).toBeNull();
  });

  test("flags an empty description", () => {
    const err = getSkillUploadError(md("name: ok_skill\ndescription:"));
    expect(err).toContain("non-empty name and description");
  });
});
