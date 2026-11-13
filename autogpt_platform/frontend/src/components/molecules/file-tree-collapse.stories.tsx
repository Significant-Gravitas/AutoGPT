import { Icon } from "@/components/atoms/Icon/Icon";
import { UnfoldLessIcon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import {
  CollapseButton,
  File,
  Folder,
  Tree,
  type TreeViewElement,
} from "./file-tree";

const ELEMENTS: TreeViewElement[] = [
  {
    id: "src",
    name: "src",
    children: [
      { id: "app", name: "app", children: [{ id: "page", name: "page.tsx" }] },
      {
        id: "components",
        name: "components",
        children: [
          { id: "button", name: "Button.tsx" },
          { id: "input", name: "Input.tsx" },
        ],
      },
      { id: "utils", name: "utils.ts" },
    ],
  },
  { id: "package", name: "package.json" },
  { id: "readme", name: "README.md" },
];

interface DemoProps {
  initialExpandedItems: string[];
}

function FileTreeWithCollapse({ initialExpandedItems }: DemoProps) {
  return (
    <div className="h-80 w-72 rounded-medium border border-separator bg-surface py-2">
      <Tree elements={ELEMENTS} initialExpandedItems={initialExpandedItems}>
        <Folder value="src" element="src">
          <Folder value="app" element="app">
            <File value="page">page.tsx</File>
          </Folder>
          <Folder value="components" element="components">
            <File value="button">Button.tsx</File>
            <File value="input">Input.tsx</File>
          </Folder>
          <File value="utils">utils.ts</File>
        </Folder>
        <File value="package">package.json</File>
        <File value="readme">README.md</File>
        <CollapseButton elements={ELEMENTS}>
          <Icon icon={UnfoldLessIcon} className="size-4" />
        </CollapseButton>
      </Tree>
    </div>
  );
}

const meta = {
  title: "Molecules/FileTree collapse",
  component: FileTreeWithCollapse,
  parameters: { layout: "centered" },
} satisfies Meta<typeof FileTreeWithCollapse>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Every folder open; the collapse button closes them all. */
export const Expanded: Story = {
  args: { initialExpandedItems: ["src", "app", "components"] },
};

/** Every folder closed; the same button opens them all. */
export const Collapsed: Story = {
  args: { initialExpandedItems: [] },
};
