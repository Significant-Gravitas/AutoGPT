'use client';

import { Close, Content, Description, Overlay, Portal, Root, Title, Trigger } from '@radix-ui/react-dialog';
import { XIcon } from 'lucide-react';
import { ComponentProps } from 'react';
import { cn } from '../../lib/utils';

/**
 * The root Dialog component that manages the state and accessibility of the dialog
 * @see https://ui.shadcn.com/docs/components/dialog
 * @returns A Dialog root component
 */
function Dialog({ ...props }: ComponentProps<typeof Root>) {
  return <Root data-slot='dialog' {...props} />;
}

/**
 * The button that opens the dialog when clicked
 * @returns A button component that triggers the dialog
 */
function DialogTrigger({ ...props }: ComponentProps<typeof Trigger>) {
  return <Trigger data-slot='dialog-trigger' {...props} />;
}

/**
 * Portal component that renders the dialog content in a portal
 * @returns A portal component for dialog content
 */

function DialogPortal({ ...props }: ComponentProps<typeof Portal>) {
  return <Portal data-slot='dialog-portal' {...props} />;
}

/**
 * Button component for closing the dialog
 * @returns A close button component
 */
function DialogClose({ ...props }: ComponentProps<typeof Close>) {
  return <Close data-slot='dialog-close' {...props} />;
}

/**
 * The overlay that covers the screen behind the dialog
 * @param props - Props for the overlay component including className and ref
 * @param props.className - Additional CSS classes to apply to the dialog overlay
 * @returns A semi-transparent overlay component
 */
function DialogOverlay({ className, ...props }: ComponentProps<typeof Overlay>) {
  return (
    <Overlay
      data-slot='dialog-overlay'
      className={cn(
        'data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 fixed inset-0 z-50 bg-black/50',
        className,
      )}
      {...props}
    />
  );
}

/**
 * The main content container of the dialog
 * @param children - The content to be displayed inside the dialog
 * @param showCloseButton- Whether to show the close button in the dialog
 * @param props - Props for the content component including className, children and ref
 * @param props.className - Additional CSS classes to apply to the dialog content
 * @returns A dialog content container component
 */
function DialogContent({
  className,
  children,
  showCloseButton = true,
  ...props
}: ComponentProps<typeof Content> & {
  showCloseButton?: boolean;
}) {
  return (
    <DialogPortal data-slot='dialog-portal'>
      <DialogOverlay />
      <Content
        data-slot='dialog-content'
        className={cn(
          'bg-background data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 fixed top-[50%] left-[50%] z-50 grid w-full max-w-[calc(100%-2rem)] translate-x-[-50%] translate-y-[-50%] gap-4 rounded-lg border p-6 shadow-lg duration-200 sm:max-w-lg',
          className,
        )}
        {...props}
      >
        {children}
        {showCloseButton && (
          <Close
            data-slot='dialog-close'
            className="ring-offset-background focus:ring-ring data-[state=open]:bg-accent data-[state=open]:text-muted-foreground absolute top-4 right-4 rounded-xs opacity-70 transition-opacity hover:opacity-100 focus:ring-2 focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
          >
            <XIcon />
            <span className='sr-only'>Close</span>
          </Close>
        )}
      </Content>
    </DialogPortal>
  );
}

/**
 * Container for the dialog header content
 * @param props - HTML div element attributes including className
 * @param props.className - Additional CSS classes to apply to the dialog header
 * @returns A header container component
 */
function DialogHeader({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      data-slot='dialog-header'
      className={cn('flex flex-col gap-2 text-center sm:text-left', className)}
      {...props}
    />
  );
}

/**
 * Container for the dialog footer content
 * @param props - HTML div element attributes including className
 * @param props.className - Additional CSS classes to apply to the dialog footer
 * @returns A footer container component
 */
function DialogFooter({ className, ...props }: ComponentProps<'div'>) {
  return (
    <div
      data-slot='dialog-footer'
      className={cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)}
      {...props}
    />
  );
}

/**
 * The title component of the dialog
 * @param props - Props for the title component including className and ref
 * @param props.className - Additional CSS classes to apply to the dialog title
 * @returns A title component for the dialog
 */
function DialogTitle({ className, ...props }: ComponentProps<typeof Title>) {
  return <Title data-slot='dialog-title' className={cn('text-lg leading-none font-semibold', className)} {...props} />;
}

/**
 * The description component of the dialog
 * @param props - Props for the description component including className and ref
 * @param props.className - Additional CSS classes to apply to the dialog description
 * @returns A description component for the dialog
 */

function DialogDescription({ className, ...props }: ComponentProps<typeof Description>) {
  return (
    <Description data-slot='dialog-description' className={cn('text-muted-foreground text-sm', className)} {...props} />
  );
}

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogOverlay,
  DialogPortal,
  DialogTitle,
  DialogTrigger,
};
