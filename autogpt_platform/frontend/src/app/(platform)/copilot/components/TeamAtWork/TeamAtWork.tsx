import { Text } from "@/components/atoms/Text/Text";
import { FirstJobCard } from "./components/FirstJobCard";
import { useTeamAtWork } from "./useTeamAtWork";

export function TeamAtWork() {
  const { items, isLoading } = useTeamAtWork();

  if (isLoading || items.length === 0) return null;

  return (
    <section data-testid="team-at-work" className="mt-5">
      <Text variant="body-medium" as="h2" tone="primary" className="mb-3">
        Your team
      </Text>
      <ul className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {items.map((item) => (
          <li key={item.expert_id}>
            <FirstJobCard item={item} />
          </li>
        ))}
      </ul>
    </section>
  );
}
