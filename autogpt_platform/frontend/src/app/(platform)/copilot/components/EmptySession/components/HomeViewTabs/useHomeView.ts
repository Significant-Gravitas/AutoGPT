import { parseAsStringLiteral, useQueryState } from "nuqs";

export const HOME_VIEWS = ["chat", "tasks"] as const;
export type HomeView = (typeof HOME_VIEWS)[number];

export function useHomeView() {
  return useQueryState(
    "show",
    parseAsStringLiteral(HOME_VIEWS).withDefault("chat"),
  );
}
