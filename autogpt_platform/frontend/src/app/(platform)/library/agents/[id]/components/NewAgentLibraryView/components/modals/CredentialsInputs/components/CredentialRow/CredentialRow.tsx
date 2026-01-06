import { IconKey } from "@/components/__legacy__/ui/icons";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { CaretDown, DotsThreeVertical } from "@phosphor-icons/react";
import {
  fallbackIcon,
  getCredentialDisplayName,
  MASKED_KEY_LENGTH,
  providerIcons,
} from "../../helpers";

type CredentialRowProps = {
  credential: {
    id: string;
    title?: string;
    username?: string;
    type: string;
    provider: string;
  };
  provider: string;
  displayName: string;
  onSelect: () => void;
  onDelete: () => void;
  readOnly?: boolean;
  showCaret?: boolean;
  asSelectTrigger?: boolean;
};

export function CredentialRow({
  credential,
  provider,
  displayName,
  onSelect,
  onDelete,
  readOnly = false,
  showCaret = false,
  asSelectTrigger = false,
}: CredentialRowProps) {
  const ProviderIcon = providerIcons[provider] || fallbackIcon;

  return (
    <div
      className={cn(
        "flex items-center gap-3 rounded-medium border border-zinc-200 bg-white p-3 transition-colors",
        asSelectTrigger ? "border-0 bg-transparent" : readOnly ? "w-fit" : "",
      )}
      onClick={readOnly || showCaret || asSelectTrigger ? undefined : onSelect}
      style={
        readOnly || showCaret || asSelectTrigger
          ? { cursor: showCaret || asSelectTrigger ? "pointer" : "default" }
          : undefined
      }
    >
      <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-gray-900">
        <ProviderIcon className="h-3 w-3 text-white" />
      </div>
      <IconKey className="h-5 w-5 shrink-0 text-zinc-800" />
      <div className="flex min-w-0 flex-1 flex-nowrap items-center gap-4">
        <Text
          variant="body"
          className="line-clamp-1 flex-[0_0_50%] text-ellipsis tracking-tight"
        >
          {getCredentialDisplayName(credential, displayName)}
        </Text>
        <Text
          variant="large"
          className="lex-[0_0_40%] relative top-1 hidden overflow-hidden whitespace-nowrap font-mono tracking-tight md:block"
        >
          {"*".repeat(MASKED_KEY_LENGTH)}
        </Text>
      </div>
      {showCaret && !asSelectTrigger && (
        <CaretDown className="h-4 w-4 shrink-0 text-gray-400" />
      )}
      {!readOnly && !showCaret && !asSelectTrigger && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="ml-auto shrink-0 rounded p-1 hover:bg-gray-100"
              onClick={(e) => e.stopPropagation()}
            >
              <DotsThreeVertical className="h-5 w-5 text-gray-400" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={(e) => {
                e.stopPropagation();
                onDelete();
              }}
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  );
}
