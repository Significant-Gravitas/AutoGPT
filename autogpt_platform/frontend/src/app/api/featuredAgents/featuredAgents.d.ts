export interface FeaturedAgent {
  agentName: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
}

export type FeaturedAgents = FeaturedAgent[];
