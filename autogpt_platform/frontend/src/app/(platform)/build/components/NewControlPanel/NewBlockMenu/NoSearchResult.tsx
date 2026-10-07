import { Sad01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

export const NoSearchResult = () => {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center text-center">
      <Icon icon={Sad01Icon} size={64} className="mb-10 text-zinc-400" />
      <div className="space-y-1">
        <Text variant="body-medium" className="text-zinc-800">
          No match found
        </Text>
        <Text variant="body" tone="secondary">
          Try adjusting your search terms
        </Text>
      </div>
    </div>
  );
};
