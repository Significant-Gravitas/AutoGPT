interface StoreListing {
  agentName: string;
  agentImage: string;
  creatorName: string;
  description: string;
  runs: number;
  rating: number;
  avatarSrc?: string;
  categories?: string[];
  lastUpdated?: Date;
  version?: string;
  mediaUrls?: string[];
}
