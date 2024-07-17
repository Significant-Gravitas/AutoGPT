export type ObjectSchema = {
    type: string;
    properties: { [key: string]: any };
    additionalProperties?: { type: string };
    required?: string[];
  };
