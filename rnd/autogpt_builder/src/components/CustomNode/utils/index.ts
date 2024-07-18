import { BlockSchema } from '@/lib/types';

export const hasOptionalFields = (schema: BlockSchema): boolean => {
  return schema && Object.keys(schema.properties).some((key) => {
    return !(schema.required?.includes(key));
  });
};

export const getValue = (key: string, hardcodedValues: { [key: string]: any }): any => {
  const keys = key.split('.');
  return keys.reduce((acc, k) => (acc && acc[k] !== undefined) ? acc[k] : '', hardcodedValues);
};

export const isHandleConnected = (key: string, connections: any[], id: string): boolean => {
  return connections && connections.some((conn: any) => {
    if (typeof conn === 'string') {
      const [source, target] = conn.split(' -> ');
      return target.includes(key) && target.includes(id);
    }
    return conn.target === id && conn.targetHandle === key;
  });
};
