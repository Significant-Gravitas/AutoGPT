/** Two rows of two; past this the header offers "Browse all skills". */
export const SHELF_SIZE = 4;

/** One page of the browse-all grid. */
export const BROWSE_PAGE_SIZE = 20;

/** Categories are stored as slugs ("content", "lead-gen"); a skill's own title
 *  is not, and comes off the API as `title`. */
export function formatCategoryLabel(category: string): string {
  const words = category.replace(/[-_]+/g, " ").trim();
  if (!words) return category;
  return words.charAt(0).toUpperCase() + words.slice(1);
}
