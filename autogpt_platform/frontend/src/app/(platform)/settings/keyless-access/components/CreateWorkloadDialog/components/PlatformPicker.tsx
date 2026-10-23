import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import { PLATFORM_OPTIONS, type Platform } from "../../../helpers";

interface Props {
  onPick: (platform: Platform) => void;
}

export function PlatformPicker({ onPick }: Props) {
  return (
    <div className="flex flex-col gap-4 px-1">
      <Text variant="body" className="text-zinc-700">
        AutoGPT will accept the identity tokens this platform gives your code,
        but only for the repository, project or server you name next.
      </Text>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {PLATFORM_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onPick(option.value)}
            className="flex items-start gap-3 rounded-lg border border-zinc-200 p-3 text-left transition-colors hover:border-zinc-400 hover:bg-zinc-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800"
          >
            <Icon
              icon={option.icon}
              size={20}
              className="mt-0.5 text-zinc-700"
            />
            <span className="flex flex-col gap-0.5">
              <Text variant="body-medium" as="span" className="text-textBlack">
                {option.label}
              </Text>
              <Text variant="small" as="span" className="text-zinc-600">
                {option.description}
              </Text>
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
