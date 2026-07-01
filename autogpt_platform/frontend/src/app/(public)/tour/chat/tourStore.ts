import { create } from "zustand";
import { TOUR_SESSION_ID } from "./script/tourChats";

interface TourState {
  activeSessionId: string;
  setActiveSession: (id: string) => void;
}

export const useTourStore = create<TourState>((set) => ({
  activeSessionId: TOUR_SESSION_ID,
  setActiveSession: (id) => set({ activeSessionId: id }),
}));
