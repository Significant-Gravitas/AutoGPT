import type { ExecutorMachine } from "@/app/api/__generated__/models/executorMachine";

export const WORKSTATION: ExecutorMachine = {
  machine_id: "1c49dced-6256-4f28-03af-f8f1ecf3eded",
  connection_id: "connection-1",
  display_name: "Workstation",
  platform: "windows",
  arch: "x86_64",
  shim_version: "0.2.0",
  capabilities: ["files"],
};
export const LAPTOP: ExecutorMachine = {
  ...WORKSTATION,
  machine_id: "36d78179-05b7-7f11-32dd-34f3999eceeb",
  connection_id: "connection-2",
  display_name: "Laptop",
  platform: "darwin",
  arch: "arm64",
};

export const STUDIO: ExecutorMachine = {
  ...LAPTOP,
  machine_id: "2e6b6f28-358a-8d25-6113-ef09e4dcce45",
  connection_id: "connection-3",
  display_name: "Studio",
};
