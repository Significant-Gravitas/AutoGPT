"use client";

import React, {
  createContext,
  forwardRef,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { cn } from "@/lib/utils";
import { Button } from "@/components/atoms/Button/Button";
import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";
import {
  FileEmpty02Icon,
  Folder01Icon,
  FolderOpenIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type TreeViewElement = {
  id: string;
  name: string;
  isSelectable?: boolean;
  children?: TreeViewElement[];
};

type TreeContextProps = {
  selectedId: string | undefined;
  expandedItems: string[] | undefined;
  indicator: boolean;
  handleExpand: (id: string) => void;
  selectItem: (id: string) => void;
  setExpandedItems?: React.Dispatch<React.SetStateAction<string[] | undefined>>;
  openIcon?: React.ReactNode;
  closeIcon?: React.ReactNode;
  direction: "rtl" | "ltr";
};

const TreeContext = createContext<TreeContextProps | null>(null);

const useTree = () => {
  const context = useContext(TreeContext);
  if (!context) {
    throw new Error("useTree must be used within a TreeProvider");
  }
  return context;
};

type Direction = "rtl" | "ltr" | undefined;

type TreeViewProps = {
  initialSelectedId?: string;
  indicator?: boolean;
  elements?: TreeViewElement[];
  initialExpandedItems?: string[];
  openIcon?: React.ReactNode;
  closeIcon?: React.ReactNode;
} & React.HTMLAttributes<HTMLDivElement>;

const Tree = forwardRef<HTMLDivElement, TreeViewProps>(
  (
    {
      className,
      elements,
      initialSelectedId,
      initialExpandedItems,
      children,
      indicator = true,
      openIcon,
      closeIcon,
      dir,
      ...props
    },
    ref,
  ) => {
    const [selectedId, setSelectedId] = useState<string | undefined>(
      initialSelectedId,
    );
    const [expandedItems, setExpandedItems] = useState<string[] | undefined>(
      initialExpandedItems,
    );

    const selectItem = useCallback((id: string) => {
      setSelectedId(id);
    }, []);

    const handleExpand = useCallback((id: string) => {
      setExpandedItems((prev) => {
        if (prev?.includes(id)) {
          return prev.filter((item) => item !== id);
        }
        return [...(prev ?? []), id];
      });
    }, []);

    const expandSpecificTargetedElements = useCallback(
      (elements?: TreeViewElement[], selectId?: string) => {
        if (!elements || !selectId) return;
        const findParent = (
          currentElement: TreeViewElement,
          currentPath: string[] = [],
        ) => {
          const isSelectable = currentElement.isSelectable ?? true;
          const newPath = [...currentPath, currentElement.id];
          if (currentElement.id === selectId) {
            if (isSelectable) {
              setExpandedItems((prev) => [...(prev ?? []), ...newPath]);
            } else {
              if (newPath.includes(currentElement.id)) {
                newPath.pop();
                setExpandedItems((prev) => [...(prev ?? []), ...newPath]);
              }
            }
            return;
          }
          if (
            isSelectable &&
            currentElement.children &&
            currentElement.children.length > 0
          ) {
            currentElement.children.forEach((child) => {
              findParent(child, newPath);
            });
          }
        };
        elements.forEach((element) => {
          findParent(element);
        });
      },
      [],
    );

    useEffect(() => {
      if (initialSelectedId) {
        expandSpecificTargetedElements(elements, initialSelectedId);
      }
    }, [initialSelectedId, elements]);

    const direction = dir === "rtl" ? "rtl" : "ltr";

    return (
      <TreeContext.Provider
        value={{
          selectedId,
          expandedItems,
          handleExpand,
          selectItem,
          setExpandedItems,
          indicator,
          openIcon,
          closeIcon,
          direction,
        }}
      >
        <div data-slot="file-tree" className={cn("size-full", className)}>
          <ScrollArea
            ref={ref}
            data-slot="file-tree-scroll-area"
            className="relative h-full px-2"
            dir={dir as Direction}
          >
            <AccordionPrimitive.Root
              data-slot="file-tree-list"
              {...props}
              type="multiple"
              defaultValue={expandedItems}
              value={expandedItems}
              className="flex flex-col gap-1"
              onValueChange={(value) =>
                setExpandedItems((prev) => [...(prev ?? []), value[0]])
              }
              dir={dir as Direction}
            >
              {children}
            </AccordionPrimitive.Root>
          </ScrollArea>
        </div>
      </TreeContext.Provider>
    );
  },
);

Tree.displayName = "Tree";

const TreeIndicator = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => {
  const { direction } = useTree();

  return (
    <div
      dir={direction}
      ref={ref}
      data-slot="file-tree-indicator"
      // TODO(tokens): no role for neutral-400 (#A3A3A3) or neutral-600
      // (#525252), nor for the neutral-100 (#F5F5F5) selected rows below.
      className={cn(
        "absolute left-1.5 h-full w-px rounded-md bg-neutral-400 py-3 duration-300 ease-in-out hover:bg-neutral-600 rtl:right-1.5",
        className,
      )}
      {...props}
    />
  );
});

TreeIndicator.displayName = "TreeIndicator";

type FolderProps = {
  expandedItems?: string[];
  element: string;
  isSelectable?: boolean;
  isSelect?: boolean;
} & React.ComponentPropsWithoutRef<typeof AccordionPrimitive.Item>;

const Folder = forwardRef<
  HTMLDivElement,
  FolderProps & React.HTMLAttributes<HTMLDivElement>
>(
  (
    {
      className,
      element,
      value,
      isSelectable = true,
      isSelect,
      children,
      ...props
    },
    _ref,
  ) => {
    const {
      direction,
      handleExpand,
      expandedItems,
      indicator,
      setExpandedItems,
      openIcon,
      closeIcon,
    } = useTree();

    return (
      <AccordionPrimitive.Item
        data-slot="file-tree-folder"
        {...props}
        value={value}
        className="relative h-full overflow-hidden"
      >
        <AccordionPrimitive.Trigger
          data-slot="file-tree-folder-trigger"
          className={cn(
            `flex items-center gap-1 rounded-md text-body leading-5`,
            className,
            {
              "rounded-md bg-neutral-100": isSelect && isSelectable,
              "cursor-pointer": isSelectable,
              "cursor-not-allowed opacity-50": !isSelectable,
            },
          )}
          disabled={!isSelectable}
          onClick={() => handleExpand(value)}
        >
          {expandedItems?.includes(value)
            ? (openIcon ?? (
                <Icon
                  data-slot="file-tree-folder-icon"
                  icon={FolderOpenIcon}
                  className="size-4"
                />
              ))
            : (closeIcon ?? (
                <Icon
                  data-slot="file-tree-folder-icon"
                  icon={Folder01Icon}
                  className="size-4"
                />
              ))}
          <span data-slot="file-tree-folder-label">{element}</span>
        </AccordionPrimitive.Trigger>
        <AccordionPrimitive.Content
          data-slot="file-tree-folder-panel"
          className="relative h-full overflow-hidden text-body leading-5 data-[state=closed]:animate-accordion-up data-[state=open]:animate-accordion-down"
        >
          {element && indicator && <TreeIndicator aria-hidden="true" />}
          <AccordionPrimitive.Root
            data-slot="file-tree-folder-list"
            dir={direction}
            type="multiple"
            className="ml-5 flex flex-col gap-1 py-1 rtl:mr-5"
            defaultValue={expandedItems}
            value={expandedItems}
            onValueChange={(value) => {
              setExpandedItems?.((prev) => [...(prev ?? []), value[0]]);
            }}
          >
            {children}
          </AccordionPrimitive.Root>
        </AccordionPrimitive.Content>
      </AccordionPrimitive.Item>
    );
  },
);

Folder.displayName = "Folder";

const File = forwardRef<
  HTMLButtonElement,
  {
    value: string;
    handleSelect?: (id: string) => void;
    isSelectable?: boolean;
    isSelect?: boolean;
    fileIcon?: React.ReactNode;
  } & React.ButtonHTMLAttributes<HTMLButtonElement>
>(
  (
    {
      value,
      className,
      handleSelect: _handleSelect,
      isSelectable = true,
      isSelect,
      fileIcon,
      children,
      ...props
    },
    ref,
  ) => {
    const { direction, selectedId, selectItem } = useTree();
    const isSelected = isSelect ?? selectedId === value;
    return (
      <button
        ref={ref}
        type="button"
        disabled={!isSelectable}
        data-slot="file-tree-file"
        className={cn(
          "flex w-fit items-center gap-1 rounded-md pr-1 text-body leading-5 duration-200 ease-in-out rtl:pl-1 rtl:pr-0",
          {
            "bg-neutral-100": isSelected && isSelectable,
          },
          isSelectable ? "cursor-pointer" : "cursor-not-allowed opacity-50",
          direction === "rtl" ? "rtl" : "ltr",
          className,
        )}
        onClick={() => selectItem(value)}
        {...props}
      >
        {fileIcon ?? (
          <Icon
            data-slot="file-tree-file-icon"
            icon={FileEmpty02Icon}
            className="size-4"
          />
        )}
        {children}
      </button>
    );
  },
);

File.displayName = "File";

const CollapseButton = forwardRef<
  HTMLButtonElement,
  {
    elements: TreeViewElement[];
    expandAll?: boolean;
  } & React.HTMLAttributes<HTMLButtonElement>
>(
  (
    { className: _className, elements, expandAll = false, children, ...props },
    ref,
  ) => {
    const { expandedItems, setExpandedItems } = useTree();

    const expendAllTree = useCallback((elements: TreeViewElement[]) => {
      const expandTree = (element: TreeViewElement) => {
        const isSelectable = element.isSelectable ?? true;
        if (isSelectable && element.children && element.children.length > 0) {
          setExpandedItems?.((prev) => [...(prev ?? []), element.id]);
          element.children.forEach(expandTree);
        }
      };

      elements.forEach(expandTree);
    }, []);

    const closeAll = useCallback(() => {
      setExpandedItems?.([]);
    }, []);

    useEffect(() => {
      if (expandAll) {
        expendAllTree(elements);
      }
    }, [expandAll]);

    return (
      <Button
        variant="compact-ghost"
        size="compact"
        data-slot="file-tree-collapse-button"
        className="absolute bottom-1 right-2 h-8 w-fit p-1"
        onClick={
          expandedItems && expandedItems.length > 0
            ? closeAll
            : () => expendAllTree(elements)
        }
        ref={ref}
        {...props}
      >
        {children}
        <span data-slot="file-tree-collapse-button-label" className="sr-only">
          Toggle
        </span>
      </Button>
    );
  },
);

CollapseButton.displayName = "CollapseButton";

export { CollapseButton, File, Folder, Tree, type TreeViewElement };
