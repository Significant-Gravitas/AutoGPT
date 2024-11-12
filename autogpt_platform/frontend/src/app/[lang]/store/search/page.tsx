export default async function Page({
    params,
    searchParams,
  }: {
    params: { lang: string },
    searchParams: { term?: string }
  }) {
    const searchTerm = searchParams.term || ''
    return <div>Search Results for: {searchTerm}</div>
  }