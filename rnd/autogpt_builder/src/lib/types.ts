export type BlockSchema = {
  type: string;
  properties: { [key: string]: any };
  required?: string[];
  enum?: string[];
  items?: BlockSchema;
  additionalProperties?: { type: string };
  title?: string;
  description?: string;
  placeholder?: string;
  allOf?: any[];
  anyOf?: any[];
  oneOf?: any[];
};
