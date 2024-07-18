import { BlockSchema } from '@/lib/types';

export type CustomNodeData = {
  blockType: string;
  title: string;
  inputSchema: BlockSchema;
  outputSchema: BlockSchema;
  hardcodedValues: { [key: string]: any };
  setHardcodedValues: (values: { [key: string]: any }) => void;
  connections: Array<{ source: string; sourceHandle: string; target: string; targetHandle: string }>;
  isPropertiesOpen: boolean;
  status?: string;
  output_data?: any;
};

export type InputFieldProps = {
  schema: any;
  fullKey: string;
  displayKey: string;
  value: any;
  error: string | null;
  handleInputChange: (key: string, value: any) => void;
  handleInputClick: (key: string) => void;
};