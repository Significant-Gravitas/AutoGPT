import { z } from "zod";

export const createTeamSchema = z.object({
  name: z
    .string()
    .trim()
    .min(1, "Name is required")
    .max(100, "Name must be 100 characters or less"),
  description: z
    .string()
    .trim()
    .max(500, "Description must be 500 characters or less")
    .optional(),
  join_policy: z.enum(["OPEN", "PRIVATE"]),
});

export type CreateTeamFormValues = z.infer<typeof createTeamSchema>;

export const JOIN_POLICY_OPTIONS = [
  { value: "OPEN", label: "Open — anyone in the org can join" },
  { value: "PRIVATE", label: "Private — invite only" },
];
