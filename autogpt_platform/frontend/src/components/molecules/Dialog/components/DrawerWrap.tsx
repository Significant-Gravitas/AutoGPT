import { Button } from "@/components/__legacy__/ui/button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { PropsWithChildren } from "react";
import { Drawer } from "vaul";
import { DialogCtx } from "../useDialogCtx";
import { compactStyles, drawerStyles, modalStyles } from "./styles";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  testId?: string;
  title: React.ReactNode;
  handleClose: () => void;
}

export function DrawerWrap({
  children,
  title,
  variant,
  testId,
  handleClose,
  isForceOpen,
  className,
}: Props) {
  const accessibleTitle = title || "Dialog";
  const hasVisibleTitle = Boolean(title);
  const isCompact = variant === "compact";

  const closeBtn = (
    <Button
      variant="link"
      aria-label="Close"
      onClick={handleClose}
      className="!focus-visible:ring-0 p-0"
    >
      <Icon icon={Cancel01Icon} width={isCompact ? "1.25rem" : "1.5rem"} />
    </Button>
  );

  return (
    <Drawer.Portal>
      <Drawer.Overlay className={drawerStyles.overlay} />
      <Drawer.Content
        aria-describedby={undefined}
        className={cn(
          drawerStyles.content,
          isCompact && compactStyles.drawerContent,
          className,
        )}
        data-testid={testId}
        onInteractOutside={handleClose}
      >
        <div
          className={cn(
            "flex w-full shrink-0 items-center justify-between",
            hasVisibleTitle
              ? isCompact
                ? compactStyles.header
                : "pb-6"
              : "pb-0",
          )}
        >
          {hasVisibleTitle ? (
            <Drawer.Title
              className={isCompact ? compactStyles.title : drawerStyles.title}
            >
              {accessibleTitle}
            </Drawer.Title>
          ) : (
            <Drawer.Title className="sr-only">{accessibleTitle}</Drawer.Title>
          )}

          {!isForceOpen ? (
            hasVisibleTitle ? (
              closeBtn
            ) : (
              <div
                className={`${modalStyles.iconWrap} transition-colors duration-200 hover:bg-gray-200 dark:hover:bg-gray-700`}
              >
                {closeBtn}
              </div>
            )
          ) : null}
        </div>
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
          <div
            className={cn(
              "flex-1 overflow-y-auto overflow-x-hidden",
              scrollbarStyles,
            )}
          >
            {children}
          </div>
        </div>
      </Drawer.Content>
    </Drawer.Portal>
  );
}
