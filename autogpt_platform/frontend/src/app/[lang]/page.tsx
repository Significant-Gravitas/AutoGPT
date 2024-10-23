import { getDictionary } from "./dictionaries";

export default async function Page({
  params: { lang },
}: {
  params: { lang: string };
}) {
  const dict = await getDictionary(lang); // en
  return <h1>{dict.home.welcome}</h1>; // Add to Cart
}
