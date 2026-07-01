"use client";

import { TourCopilot } from "./TourCopilot";
import { useTourBootstrap } from "./useTourBootstrap";

export default function TourChatPage() {
  useTourBootstrap();
  return <TourCopilot />;
}
