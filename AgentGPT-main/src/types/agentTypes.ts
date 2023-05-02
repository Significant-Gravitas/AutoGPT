import { z } from "zod";

export const messageParser = z.object({
  type: z.enum(["goal", "thinking", "task", "action", "system"]),
  info: z.string().optional(),
  value: z.string(),
});

export type Message = z.infer<typeof messageParser>;
