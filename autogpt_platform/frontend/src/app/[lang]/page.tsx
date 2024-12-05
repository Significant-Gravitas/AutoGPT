import { redirect } from "next/navigation";

export default async function Page({
  params: { lang },
}: {
  params: { lang: string };
}) {
  redirect("/store");
}
