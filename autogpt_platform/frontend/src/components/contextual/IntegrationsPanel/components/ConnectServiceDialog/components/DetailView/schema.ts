import { z } from "zod";

/**
 * A `<input type="date">` value ("YYYY-MM-DD") as unix seconds at 23:59:59
 * local time on that day, so the key stays valid through the day picked.
 * `Date.parse` would read it as UTC midnight instead.
 */
export function dateInputToEndOfLocalDayUnix(
  value: string | undefined,
): number | undefined {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value?.trim() ?? "");
  if (!match) return undefined;
  const [year, month, day] = match.slice(1).map(Number);
  const date = new Date(year, month - 1, day, 23, 59, 59);
  if (date.getMonth() !== month - 1 || date.getDate() !== day) return undefined;
  return Math.floor(date.getTime() / 1000);
}

export const apiKeyConnectSchema = z
  .object({
    title: z
      .string()
      .trim()
      .min(1, "Name is required")
      .max(100, "Name must be 100 characters or less"),
    apiKey: z.string().trim().min(1, "API key is required"),
    expiresAt: z
      .string()
      .trim()
      .optional()
      .superRefine((value, ctx) => {
        if (!value) return;
        const expiresAt = dateInputToEndOfLocalDayUnix(value);
        if (expiresAt === undefined) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Pick a valid date",
          });
          return;
        }
        if (expiresAt * 1000 <= Date.now()) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Expiry must be in the future",
          });
        }
      }),
  })
  .strict();

export type ApiKeyConnectFormValues = z.infer<typeof apiKeyConnectSchema>;
