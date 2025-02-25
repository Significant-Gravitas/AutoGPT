"use server";
import BackendAPI from "@/lib/autogpt-server-api";
import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";

export async function runGraph(id: string, version: number, inputData?: {
  [key: string]: any;
}) {
  const api = new BackendAPI();
  api.executeGraph(id, version, inputData);
  revalidatePath("/monitoring", "layout");
  redirect("/monitoring");
}
