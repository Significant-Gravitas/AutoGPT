import { parseAsStringLiteral, useQueryState } from "nuqs";

const tabParser = parseAsStringLiteral([
  "basics",
  "schedules",
  "workflows",
  "skills",
] as const)
  .withDefault("basics")
  .withOptions({ history: "push" });

export function useAutopilotTab() {
  const [activeTab, setActiveTab] = useQueryState("tab", tabParser);

  function onTabChange(value: string) {
    void setActiveTab(tabParser.parse(value));
  }

  return { activeTab, onTabChange };
}
