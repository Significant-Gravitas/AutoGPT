"use client";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ArrowRight01Icon, Home01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

export interface BreadcrumbItem {
  id: string;
  name: string;
}

interface Props {
  /** Ancestors first, the open folder last. */
  items: BreadcrumbItem[];
  /** `null` navigates to the root. */
  onNavigate: (folderId: string | null) => void;
  /** Compact type and spacing, for the file picker inside a dialog. */
  compact?: boolean;
}

// `⌂ Files › Reports › 2026 › Q3` is the widest chain shown whole; past that
// everything between the root and the parent collapses behind the "…" menu,
// so the root, the parent and the current folder always stay visible.
const MAX_CRUMBS_BESIDE_ROOT = 3;

export function FolderBreadcrumb({ items, onNavigate, compact }: Props) {
  const { hidden, visible } = splitForOverflow(items);
  const separator = (
    <Icon
      icon={ArrowRight01Icon}
      size={compact ? 12 : 14}
      className="shrink-0 text-zinc-400"
      aria-hidden
    />
  );

  return (
    <nav
      aria-label="Breadcrumb"
      data-testid="folder-breadcrumb"
      className="min-w-0"
    >
      <ol
        className={cn(
          "flex min-w-0 items-center gap-1.5 text-zinc-500",
          compact && "gap-1",
        )}
      >
        <li className="flex shrink-0 items-center">
          <button
            type="button"
            onClick={() => onNavigate(null)}
            className={CRUMB_BUTTON_CLASS}
            data-testid="folder-breadcrumb-root"
          >
            <Icon icon={Home01Icon} size={compact ? 14 : 16} />
            <Text variant="small-medium" as="span">
              Files
            </Text>
          </button>
        </li>
        {hidden.length > 0 ? (
          <li className="flex shrink-0 items-center gap-1.5">
            {separator}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  aria-label="Show hidden folders"
                  className={CRUMB_BUTTON_CLASS}
                  data-testid="folder-breadcrumb-overflow"
                >
                  <Text variant="small-medium" as="span">
                    …
                  </Text>
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-48">
                {hidden.map((item) => (
                  <DropdownMenuItem
                    key={item.id}
                    onSelect={() => onNavigate(item.id)}
                  >
                    <span className="truncate">{item.name}</span>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </li>
        ) : null}
        {visible.map((item, index) => {
          const isCurrent = index === visible.length - 1;
          return (
            <li
              key={item.id}
              className={cn(
                "flex min-w-0 items-center gap-1.5",
                isCurrent ? "min-w-0" : "shrink-0",
              )}
            >
              {separator}
              {isCurrent ? (
                <Text
                  variant="small-medium"
                  as="span"
                  aria-current="page"
                  className="max-w-[12rem] truncate text-zinc-800"
                  title={item.name}
                >
                  {item.name}
                </Text>
              ) : (
                <button
                  type="button"
                  onClick={() => onNavigate(item.id)}
                  className={CRUMB_BUTTON_CLASS}
                  title={item.name}
                >
                  <Text
                    variant="small-medium"
                    as="span"
                    className="max-w-[12rem] truncate"
                  >
                    {item.name}
                  </Text>
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}

const CRUMB_BUTTON_CLASS =
  "inline-flex min-w-0 items-center gap-1.5 rounded-md px-1.5 py-1 hover:bg-zinc-100 hover:text-zinc-800";

function splitForOverflow(items: BreadcrumbItem[]): {
  hidden: BreadcrumbItem[];
  visible: BreadcrumbItem[];
} {
  if (items.length <= MAX_CRUMBS_BESIDE_ROOT)
    return { hidden: [], visible: items };
  return { hidden: items.slice(0, -2), visible: items.slice(-2) };
}
