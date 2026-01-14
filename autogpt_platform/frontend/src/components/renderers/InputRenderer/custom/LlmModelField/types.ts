export type LlmModelMetadata = {
  creator: string;
  creator_name: string;
  title: string;
  provider: string;
  provider_name: string;
  name: string;
  price_tier?: number;
};

export type LlmModelMetadataMap = Record<string, LlmModelMetadata>;
