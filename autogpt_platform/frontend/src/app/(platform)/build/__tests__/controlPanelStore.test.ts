import { describe, it, expect, beforeEach } from "vitest";
import { useControlPanelStore } from "../stores/controlPanelStore";

beforeEach(() => {
  useControlPanelStore.getState().reset();
});

describe("controlPanelStore", () => {
  describe("initial state", () => {
    it("starts with all panels closed", () => {
      const state = useControlPanelStore.getState();
      expect(state.blockMenuOpen).toBe(false);
      expect(state.saveControlOpen).toBe(false);
      expect(state.forceOpenBlockMenu).toBe(false);
      expect(state.forceOpenSave).toBe(false);
    });
  });

  describe("setBlockMenuOpen", () => {
    it("opens the block menu", () => {
      useControlPanelStore.getState().setBlockMenuOpen(true);
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(true);
    });

    it("closes the block menu", () => {
      useControlPanelStore.getState().setBlockMenuOpen(true);
      useControlPanelStore.getState().setBlockMenuOpen(false);
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(false);
    });
  });

  describe("setSaveControlOpen", () => {
    it("opens the save control", () => {
      useControlPanelStore.getState().setSaveControlOpen(true);
      expect(useControlPanelStore.getState().saveControlOpen).toBe(true);
    });

    it("closes the save control", () => {
      useControlPanelStore.getState().setSaveControlOpen(true);
      useControlPanelStore.getState().setSaveControlOpen(false);
      expect(useControlPanelStore.getState().saveControlOpen).toBe(false);
    });
  });

  describe("setForceOpenBlockMenu", () => {
    it("sets force open state", () => {
      useControlPanelStore.getState().setForceOpenBlockMenu(true);
      expect(useControlPanelStore.getState().forceOpenBlockMenu).toBe(true);
    });

    it("does not affect blockMenuOpen", () => {
      useControlPanelStore.getState().setForceOpenBlockMenu(true);
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(false);
    });
  });

  describe("setForceOpenSave", () => {
    it("sets force open state", () => {
      useControlPanelStore.getState().setForceOpenSave(true);
      expect(useControlPanelStore.getState().forceOpenSave).toBe(true);
    });

    it("does not affect saveControlOpen", () => {
      useControlPanelStore.getState().setForceOpenSave(true);
      expect(useControlPanelStore.getState().saveControlOpen).toBe(false);
    });
  });

  describe("independent panel state", () => {
    it("opening block menu does not affect save control", () => {
      useControlPanelStore.getState().setBlockMenuOpen(true);
      expect(useControlPanelStore.getState().saveControlOpen).toBe(false);
    });

    it("opening save control does not affect block menu", () => {
      useControlPanelStore.getState().setSaveControlOpen(true);
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(false);
    });

    it("both panels can be open simultaneously", () => {
      useControlPanelStore.getState().setBlockMenuOpen(true);
      useControlPanelStore.getState().setSaveControlOpen(true);
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(true);
      expect(useControlPanelStore.getState().saveControlOpen).toBe(true);
    });
  });

  describe("reset", () => {
    it("resets all state to defaults", () => {
      useControlPanelStore.getState().setBlockMenuOpen(true);
      useControlPanelStore.getState().setSaveControlOpen(true);
      useControlPanelStore.getState().setForceOpenBlockMenu(true);
      useControlPanelStore.getState().setForceOpenSave(true);

      useControlPanelStore.getState().reset();

      const state = useControlPanelStore.getState();
      expect(state.blockMenuOpen).toBe(false);
      expect(state.saveControlOpen).toBe(false);
      expect(state.forceOpenBlockMenu).toBe(false);
      expect(state.forceOpenSave).toBe(false);
    });
  });
});
