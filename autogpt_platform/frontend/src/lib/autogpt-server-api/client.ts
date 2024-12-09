import { AutoGPTServerAPIClientSide } from "./client-client";
import { AutoGPTServerAPIServerSide } from "./client-server";

export const AutoGPTServerAPI =
  typeof window !== "undefined"
    ? AutoGPTServerAPIClientSide
    : AutoGPTServerAPIServerSide;

export type AutoGPTServerAPI =
  | AutoGPTServerAPIClientSide
  | AutoGPTServerAPIServerSide;
