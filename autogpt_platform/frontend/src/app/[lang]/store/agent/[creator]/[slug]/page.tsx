export default async function Page({
    params,
  }: {
    params: Promise<{ lang: string, creator: string, slug: string }>
  }) {
    const { lang, creator, slug } = await params
    return <div>My Post: {slug}</div>
  }