import { createSafeStorage } from "./safe-storage";

export enum SessionKey {
  CHAT_SENT_INITIAL_PROMPTS = "chat_sent_initial_prompts",
  CHAT_INITIAL_PROMPTS = "chat_initial_prompts",
}

export const sessionStorage = createSafeStorage<SessionKey>("session");
