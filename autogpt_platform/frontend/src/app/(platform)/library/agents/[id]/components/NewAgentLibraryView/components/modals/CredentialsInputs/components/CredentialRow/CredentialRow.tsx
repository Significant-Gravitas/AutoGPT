import { IconKey } from "@/components/__legacy__/ui/icons";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { CaretDownIcon, DotsThreeVertical } from "@phosphor-icons/react";
import { useEffect, useRef, useState } from "react";
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
  onDelete?: () => void;
  readOnly?: boolean;
  showCaret?: boolean;
  asSelectTrigger?: boolean;
  /** When "node", applies compact styling for node context */
  variant?: "default" | "node";
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
  variant = "default",
}: CredentialRowProps) {
  const ProviderIcon = providerIcons[provider] || fallbackIcon;
  const isNodeVariant = variant === "node";
  const containerRef = useRef<HTMLDivElement>(null);
  const [showMaskedKey, setShowMaskedKey] = useState(true);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const width = entry.contentRect.width;
        setShowMaskedKey(width >= 360);
      }
    });

    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={cn(
        "flex min-w-[20rem] items-center gap-3 rounded-medium border border-zinc-200 bg-white p-3 transition-colors",
        asSelectTrigger && isNodeVariant
          ? "min-w-0 flex-1 overflow-hidden border-0 bg-transparent"
          : asSelectTrigger
            ? "border-0 bg-transparent"
            : readOnly
              ? "w-fit"
              : "",
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
      <div
        className={cn(
          "relative flex min-w-0 flex-1 flex-nowrap items-center gap-4",
          isNodeVariant && "overflow-hidden",
        )}
      >
        <Text
          variant="body"
          className={cn(
            "min-w-0 flex-1 tracking-tight",
            isNodeVariant ? "truncate" : "line-clamp-1 text-ellipsis",
          )}
        >
          {getCredentialDisplayName(credential, displayName)}
        </Text>
        {!(asSelectTrigger && isNodeVariant) && showMaskedKey && (
          <Text
            variant="large"
            className={cn(
              "absolute top-[65%] -translate-y-1/2 overflow-hidden whitespace-nowrap font-mono tracking-tight",
              asSelectTrigger ? "right-0" : "right-6",
            )}
          >
            {"*".repeat(MASKED_KEY_LENGTH)}
          </Text>
        )}
      </div>
      {(showCaret || (asSelectTrigger && !readOnly)) && (
        <CaretDownIcon className="h-4 w-4 shrink-0 text-gray-400" />
      )}
      {!readOnly && !showCaret && !asSelectTrigger && onDelete && (
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
