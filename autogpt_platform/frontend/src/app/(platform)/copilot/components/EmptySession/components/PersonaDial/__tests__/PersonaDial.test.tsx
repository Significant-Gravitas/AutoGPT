import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PERSONAS } from "../../../personas";
import { PersonaDial } from "../PersonaDial";

afterEach(cleanup);

function renderDial(overrides?: {
  selectedIndex?: number;
  onSelect?: (index: number) => void;
  onClose?: () => void;
}) {
  const onSelect = overrides?.onSelect ?? vi.fn();
  const onClose = overrides?.onClose ?? vi.fn();
  render(
    <PersonaDial
      personas={PERSONAS}
      selectedIndex={overrides?.selectedIndex ?? 0}
      onSelect={onSelect}
      onClose={onClose}
    />,
  );
  return { onSelect, onClose };
}

describe("PersonaDial", () => {
  it("renders every persona exactly once for the full roster", () => {
    renderDial();
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(PERSONAS.length);
    const labels = options.map((o) => o.getAttribute("aria-label"));
    expect(new Set(labels).size).toBe(PERSONAS.length);
  });

  it("filters the ring as the user types, without duplicates", () => {
    renderDial();
    fireEvent.change(screen.getByLabelText("Search personas"), {
      target: { value: "byte" },
    });
    const options = screen.getAllByRole("option");
    expect(options).toHaveLength(1);
    expect(options[0].getAttribute("aria-label")).toBe("Byte — Coder");
  });

  it("shows an empty state when nothing matches", () => {
    renderDial();
    fireEvent.change(screen.getByLabelText("Search personas"), {
      target: { value: "zzz" },
    });
    expect(screen.queryAllByRole("option")).toHaveLength(0);
    expect(screen.getByText(/No personas match/)).toBeTruthy();
  });

  it("selects the first match and closes on Enter", () => {
    const { onSelect, onClose } = renderDial();
    const input = screen.getByLabelText("Search personas");
    fireEvent.change(input, { target: { value: "sage" } });
    fireEvent.keyDown(input, { key: "Enter" });
    const sageIndex = PERSONAS.findIndex((p) => p.id === "sage");
    expect(onSelect).toHaveBeenCalledWith(sageIndex);
    expect(onClose).toHaveBeenCalled();
  });

  it("steps to the neighbouring persona with the arrow buttons", () => {
    const { onSelect } = renderDial();
    fireEvent.click(screen.getByLabelText("Next persona"));
    expect(onSelect).toHaveBeenCalledWith(1);
    fireEvent.click(screen.getByLabelText("Previous persona"));
    expect(onSelect).toHaveBeenCalledWith(PERSONAS.length - 1);
  });

  it("steps with the keyboard arrows", () => {
    const { onSelect } = renderDial();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("closes on Escape", () => {
    const { onClose } = renderDial();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalled();
  });

  it("closes when the pointer goes down outside the picker", () => {
    const { onClose } = renderDial();
    fireEvent.pointerDown(document.body);
    expect(onClose).toHaveBeenCalled();
  });

  it("live-selects whichever persona reaches the bottom while dragging", () => {
    const { onSelect } = renderDial();
    const ring = document.querySelector<HTMLElement>(
      "[data-persona-picker] .touch-none",
    )!;
    // In the test DOM the ring's rect collapses to (0,0); angles still resolve.
    fireEvent.pointerDown(ring, { clientX: 100, clientY: 0, pointerId: 1 });
    fireEvent.pointerMove(ring, { clientX: 0, clientY: 100, pointerId: 1 });
    fireEvent.pointerUp(ring, { clientX: 0, clientY: 100, pointerId: 1 });
    // A 90° clockwise swing is ~3 slots backwards around an 11-slot ring.
    expect(onSelect).toHaveBeenCalledWith(8);
  });

  it("picks the persona under a tap on the rim", () => {
    const { onSelect } = renderDial();
    const ring = document.querySelector<HTMLElement>(
      "[data-persona-picker] .touch-none",
    )!;
    // Tap at the ring's bottom point (radius 380 below the collapsed centre).
    fireEvent.pointerDown(ring, { clientX: 0, clientY: 380, pointerId: 1 });
    fireEvent.pointerUp(ring, { clientX: 0, clientY: 380, pointerId: 1 });
    expect(onSelect).toHaveBeenCalledWith(0);
  });

  it("ignores taps that miss the rim", () => {
    const { onSelect, onClose } = renderDial();
    const ring = document.querySelector<HTMLElement>(
      "[data-persona-picker] .touch-none",
    )!;
    fireEvent.pointerDown(ring, { clientX: 0, clientY: 10, pointerId: 1 });
    fireEvent.pointerUp(ring, { clientX: 0, clientY: 10, pointerId: 1 });
    expect(onSelect).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("marks the selected persona for assistive tech", () => {
    renderDial({ selectedIndex: 2 });
    const selected = screen
      .getAllByRole("option")
      .filter((o) => o.getAttribute("aria-selected") === "true");
    expect(selected).toHaveLength(1);
    expect(selected[0].getAttribute("aria-label")).toBe(
      `${PERSONAS[2].name} — ${PERSONAS[2].role}`,
    );
  });
});
