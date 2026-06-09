declare module "pg" {
  export interface QueryResult<Row = unknown> {
    rows: Row[];
  }

  export class Pool {
    constructor(config?: { connectionString?: string; max?: number });

    query<Row = unknown>(
      text: string,
      values?: readonly unknown[],
    ): Promise<QueryResult<Row>>;
    end(): Promise<void>;
  }
}
