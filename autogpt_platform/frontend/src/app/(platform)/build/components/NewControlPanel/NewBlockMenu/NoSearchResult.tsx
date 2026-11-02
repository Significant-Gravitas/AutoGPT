import { Sad01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export const NoSearchResult = () => {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center text-center">
      <Icon
        icon={Sad01Icon}
        size={64}
        className="mb-10 text-foreground-subtle"
      />
      <div className="space-y-1">
        <Text variant="body-medium" tone="emphasis">
          No match found
        </Text>
        <Text variant="body" tone="secondary">
          Try adjusting your search terms
        </Text>
      </div>
    </div>
  );
};
