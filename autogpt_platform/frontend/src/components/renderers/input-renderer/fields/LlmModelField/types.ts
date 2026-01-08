export type LlmModelMetadata = {
  creator: string;
  title: string;
  provider: string;
  name: string;
  price_tier?: number;
};

export type LlmModelMetadataMap = Record<string, LlmModelMetadata>;
