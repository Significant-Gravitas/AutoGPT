import type { FeaturedAgents } from "./featuredAgents";

export async function getFeaturedAgents(): Promise<FeaturedAgents> {
  if (!process.env.NEXT_PUBLIC_AGPT_SERVER_URL) {
    throw new Error("NEXT_PUBLIC_AGPT_SERVER_URL is not set!");
  }
  try {
    const res = await fetch(
      `${process.env.NEXT_PUBLIC_AGPT_SERVER_URL}/featured-agents`,
    );

    if (!res.ok) {
      // Render the closest `error.js` Error Boundary
      throw new Error("Something went wrong while fetching featured agents!");
    }

    const featuredAgents = (await res.json()) as FeaturedAgents;
    return featuredAgents;
  } catch (error) {
    console.error("Error fetching featured agents:", error);
    return [
      {
        agentName: "Super SEO Optimizer",
        agentImage:
          "https://ddz4ak4pa3d19.cloudfront.net/cache/cc/11/cc1172271dcf723a34f488a3344e82b2.jpg",
        creatorName: "AI Labs",
        description:
          "Boost your website's search engine rankings with our advanced AI-powered SEO optimization tool.",
        runs: 100000,
        rating: 4.9,
      },
      {
        agentName: "Content Wizard",
        agentImage:
          "https://upload.wikimedia.org/wikipedia/commons/c/c5/Big_buck_bunny_poster_big.jpg",
        creatorName: "WriteRight Inc.",
        description:
          "Generate high-quality, engaging content for your blog, social media, or marketing campaigns.",
        runs: 75000,
        rating: 4.7,
      },
    ];
  }
}
