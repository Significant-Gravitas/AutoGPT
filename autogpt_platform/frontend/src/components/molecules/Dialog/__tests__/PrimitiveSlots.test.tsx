import { describe, expect, test } from "vitest";
import { render } from "@testing-library/react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "../components/DialogPrimitives";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "../components/SheetPrimitives";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/atoms/Select/components/SelectPrimitives";

// Every element in the document that carries a slot, in document order.
function slots() {
  return Array.from(document.querySelectorAll("[data-slot]")).map((el) =>
    el.getAttribute("data-slot"),
  );
}

describe("primitive slots", () => {
  test("DialogPrimitives name every element they render", () => {
    render(
      <Dialog open>
        <DialogTrigger>Open</DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Title</DialogTitle>
            <DialogDescription>Description</DialogDescription>
          </DialogHeader>
          <DialogFooter>Footer</DialogFooter>
        </DialogContent>
      </Dialog>,
    );
    expect(slots()).toEqual([
      "dialog-trigger",
      "dialog-overlay",
      "dialog-popup",
      "dialog-header",
      "dialog-title",
      "dialog-description",
      "dialog-footer",
      "dialog-close",
      "dialog-close-icon",
      "dialog-close-label",
    ]);
  });

  test("the Google-picker rule's slots sit on the elements data-dialog-* marks", () => {
    render(
      <Dialog open>
        <DialogContent>
          <DialogTitle>Title</DialogTitle>
        </DialogContent>
      </Dialog>,
    );
    const overlay = document.querySelector("[data-slot=dialog-overlay]");
    const popup = document.querySelector("[data-slot=dialog-popup]");
    expect(overlay?.hasAttribute("data-dialog-overlay")).toBe(true);
    expect(popup?.hasAttribute("data-dialog-content")).toBe(true);
    expect(popup?.getAttribute("role")).toBe("dialog");
  });

  test("SheetPrimitives name every element they render", () => {
    render(
      <Sheet open>
        <SheetTrigger>Open</SheetTrigger>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Title</SheetTitle>
            <SheetDescription>Description</SheetDescription>
          </SheetHeader>
        </SheetContent>
      </Sheet>,
    );
    expect(slots()).toEqual([
      "sheet-trigger",
      "sheet-overlay",
      "sheet-popup",
      "sheet-close",
      "sheet-close-icon",
      "sheet-close-label",
      "sheet-header",
      "sheet-title",
      "sheet-description",
    ]);
  });

  test("with asChild the trigger's child keeps its own slot", () => {
    render(
      <>
        <Dialog>
          <DialogTrigger asChild>
            <Button>Open dialog</Button>
          </DialogTrigger>
        </Dialog>
        <Sheet>
          <SheetTrigger asChild>
            <Button>Open sheet</Button>
          </SheetTrigger>
        </Sheet>
      </>,
    );
    const buttons = Array.from(document.querySelectorAll("button"));
    expect(buttons.map((b) => b.getAttribute("data-slot"))).toEqual([
      "button",
      "button",
    ]);
    expect(slots()).not.toContain("dialog-trigger");
    expect(slots()).not.toContain("sheet-trigger");
  });

  test("TablePrimitives name every element they render", () => {
    render(
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>H</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          <TableRow>
            <TableCell>C</TableCell>
          </TableRow>
        </TableBody>
      </Table>,
    );
    expect(slots()).toEqual([
      "table-container",
      "table-element",
      "table-header",
      "table-row",
      "table-head",
      "table-body",
      "table-row",
      "table-cell",
    ]);
  });

  test("a caller's data-slot wins over the primitive's default", () => {
    render(
      <Table data-slot="table-custom">
        <TableBody>
          <TableRow>
            <TableCell>x</TableCell>
          </TableRow>
        </TableBody>
      </Table>,
    );
    expect(document.querySelector("table")?.getAttribute("data-slot")).toBe(
      "table-custom",
    );
  });

  // The scroll buttons (select-scroll-up / -down and their icons) render only
  // when the open list overflows, which the test DOM cannot lay out.
  test("an open Select names every element it renders", () => {
    render(
      <Select defaultValue="a" open>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="a">A</SelectItem>
            <SelectSeparator />
            <SelectItem value="b">B</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>,
    );
    expect(slots()).toEqual([
      "select-trigger",
      "select-value",
      "select-icon",
      "select-popup",
      "select-viewport",
      "select-group",
      "select-item",
      "select-item-indicator-wrap",
      "select-item-indicator",
      "select-item-check-icon",
      "select-item-text",
      "select-separator",
      "select-item",
      "select-item-indicator-wrap",
      "select-item-text",
    ]);
  });
});
