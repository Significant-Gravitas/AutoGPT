"use server";

import { TutorialStepData } from "@/lib/autogpt-server-api/types";
import AutoGPTServerAPIServer from "@/lib/autogpt-server-api/clientServer";

export const sendTutorialStep = async (data: TutorialStepData) => {
  console.debug("sendTutorialStep", data);
  const api = new AutoGPTServerAPIServer();

  await api.logTutorialStep(data);
};
