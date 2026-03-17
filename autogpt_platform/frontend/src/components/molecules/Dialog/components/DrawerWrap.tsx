import { Button } from "@/components/__legacy__/ui/button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";
import { X } from "@phosphor-icons/react";
import { PropsWithChildren } from "react";
import { Drawer } from "vaul";
import { DialogCtx } from "../useDialogCtx";
import { drawerStyles, modalStyles } from "./styles";

type BaseProps = DialogCtx & PropsWithChildren;

interface Props extends BaseProps {
  testId?: string;
  title: React.ReactNode;
  handleClose: () => void;
}

export function DrawerWrap({
  children,
  title,
  testId,
  handleClose,
  isForceOpen,
}: Props) {
  const accessibleTitle = title ?? "Dialog";
  const hasVisibleTitle = Boolean(title);

  const closeBtn = (
    <Button
      variant="link"
      aria-label="Close"
      onClick={handleClose}
      className="!focus-visible:ring-0 p-0"
    >
      <X width="1.5rem" />
    </Button>
  );

  return (
    <Drawer.Portal>
      <Drawer.Overlay className={drawerStyles.overlay} />
      <Drawer.Content
        aria-describedby={undefined}
        className={drawerStyles.content}
        data-testid={testId}
        onInteractOutside={handleClose}
      >
        <div
          className={`flex w-full shrink-0 items-center justify-between ${
            hasVisibleTitle ? "pb-6" : "pb-0"
          }`}
        >
          {hasVisibleTitle ? (
            <Drawer.Title className={drawerStyles.title}>
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
