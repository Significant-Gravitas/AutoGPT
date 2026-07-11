import { z } from "zod";

// Mirrors the slug conventions of CreateOrgDialog's slug field so every org
// slug in the app follows the same rules. The backend accepts a slightly wider
// range, but reusing these keeps aliases consistent with the primary slug.
export const createAliasSchema = z.object({
  alias_slug: z
    .string()
    .trim()
    .min(3, "Slug must be at least 3 characters")
    .max(50, "Slug must be 50 characters or less")
    .regex(
      /^[a-z0-9]+(?:-[a-z0-9]+)*$/,
      "Lowercase letters, numbers and dashes only",
    ),
});

export type CreateAliasFormValues = z.infer<typeof createAliasSchema>;
