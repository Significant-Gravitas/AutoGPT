import type { Meta } from "@storybook/nextjs";
import type { ComponentProps } from "react";
import { Tree, Folder, File, type TreeViewElement } from "./file-tree";

const meta: Meta<typeof Tree> = {
  title: "Molecules/FileTree",
  component: Tree,
  parameters: {
    layout: "centered",
  },
};

export default meta;

const SIMPLE_ELEMENTS: TreeViewElement[] = [
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

function SimpleFileTree(props: Omit<ComponentProps<typeof Tree>, "children">) {
  return (
    <Tree {...props}>
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
    </Tree>
  );
}

export function Default() {
  return (
    <div className="w-72">
      <SimpleFileTree
        elements={SIMPLE_ELEMENTS}
        initialExpandedItems={["src"]}
      />
    </div>
  );
}

export function AllExpanded() {
  return (
    <div className="w-72">
      <SimpleFileTree
        elements={SIMPLE_ELEMENTS}
        initialExpandedItems={["src", "app", "components"]}
      />
    </div>
  );
}

export function FoldersOnly() {
  return (
    <div className="w-72">
      <Tree
        initialExpandedItems={["marketing", "engineering"]}
        elements={[
          {
            id: "marketing",
            name: "Marketing",
            children: [{ id: "social", name: "Social Media" }],
          },
          {
            id: "engineering",
            name: "Engineering",
            children: [
              { id: "backend", name: "Backend" },
              { id: "frontend", name: "Frontend" },
            ],
          },
          { id: "sales", name: "Sales" },
        ]}
      >
        <Folder value="marketing" element="Marketing">
          <Folder value="social" element="Social Media" />
        </Folder>
        <Folder value="engineering" element="Engineering">
          <Folder value="backend" element="Backend" />
          <Folder value="frontend" element="Frontend" />
        </Folder>
        <Folder value="sales" element="Sales" />
      </Tree>
    </div>
  );
}

export function WithInitialSelection() {
  return (
    <div className="w-72">
      <SimpleFileTree
        elements={SIMPLE_ELEMENTS}
        initialSelectedId="button"
        initialExpandedItems={["src", "components"]}
      />
    </div>
  );
}

export function NoIndicator() {
  return (
    <div className="w-72">
      <SimpleFileTree
        elements={SIMPLE_ELEMENTS}
        indicator={false}
        initialExpandedItems={["src", "components"]}
      />
    </div>
  );
}
