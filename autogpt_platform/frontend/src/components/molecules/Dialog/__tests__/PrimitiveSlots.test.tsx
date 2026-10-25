import { describe, expect, test } from "vitest";
import { render } from "@testing-library/react";
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

function slots() {
  return Array.from(document.querySelectorAll("[data-slot]")).map((el) =>
    el.getAttribute("data-slot"),
  );
}

describe("primitive slots", () => {
  test("DialogPrimitives name every element, and the picker rule's slots sit on the overlay and popup", () => {
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
    expect(slots()).toEqual(
      expect.arrayContaining([
        "dialog-trigger",
        "dialog-overlay",
        "dialog-popup",
        "dialog-header",
        "dialog-title",
        "dialog-description",
        "dialog-footer",
        "dialog-close",
        "dialog-close-icon",
      ]),
    );
    const overlay = document.querySelector("[data-slot=dialog-overlay]");
    const popup = document.querySelector("[data-slot=dialog-popup]");
    expect(overlay?.hasAttribute("data-dialog-overlay")).toBe(true);
    expect(popup?.hasAttribute("data-dialog-content")).toBe(true);
    expect(popup?.getAttribute("role")).toBe("dialog");
  });

  test("a caller's data-slot still wins over the primitive's default", () => {
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

  test("SheetPrimitives name every element", () => {
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
    expect(slots()).toEqual(
      expect.arrayContaining([
        "sheet-trigger",
        "sheet-overlay",
        "sheet-popup",
        "sheet-close",
        "sheet-close-icon",
        "sheet-header",
        "sheet-title",
        "sheet-description",
      ]),
    );
  });

  test("TablePrimitives name every element", () => {
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

  test("SelectPrimitives name the trigger, value and icon", () => {
    render(
      <Select defaultValue="a">
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectItem value="a">A</SelectItem>
            <SelectSeparator />
          </SelectGroup>
        </SelectContent>
      </Select>,
    );
    expect(slots()).toEqual(
      expect.arrayContaining(["select-trigger", "select-value", "select-icon"]),
    );
  });
});
