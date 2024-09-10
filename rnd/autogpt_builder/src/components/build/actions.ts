"use server";

import { TutorialStepData } from "@/lib/autogpt-server-api/types";
import AutoGPTServerAPI from "@/lib/autogpt-server-api/client";

export const sendTutorialStep = async (data: TutorialStepData) => {
  console.log("sendTutorialStep", data);
  const api = new AutoGPTServerAPI();

  await api.logTutorialStep(data);
};
