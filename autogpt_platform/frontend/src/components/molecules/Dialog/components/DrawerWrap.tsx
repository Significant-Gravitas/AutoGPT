import { Button } from "@/components/ui/button";
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
  const closeBtn = (
    <Button variant="link" aria-label="Close" onClick={handleClose}>
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
          className={`flex w-full items-center justify-between ${
            title ? "pb-6" : "pb-0"
          }`}
        >
          {title ? (
            <Drawer.Title className={drawerStyles.title}>{title}</Drawer.Title>
          ) : null}

          {!isForceOpen ? (
            title ? (
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
        <div className="overflow-auto">{children}</div>
      </Drawer.Content>
    </Drawer.Portal>
  );
}
