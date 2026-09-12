/** Two rows of two; past this the header offers "Browse all skills". */
export const SHELF_SIZE = 4;

/** One page of the browse-all grid. */
export const BROWSE_PAGE_SIZE = 20;

/** The API returns a skill's frontmatter name, which the seed pins to the slug
 *  (`skill_seed.py`), so production has "outreach-playbook" where a title
 *  belongs. Humanise it until the model carries a display title. */
export function formatSkillTitle(name: string): string {
  const words = name.replace(/[-_]+/g, " ").trim();
  if (!words) return name;
  return words.charAt(0).toUpperCase() + words.slice(1);
}
