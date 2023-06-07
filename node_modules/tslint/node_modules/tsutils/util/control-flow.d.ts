import * as ts from 'typescript';
export declare function endsControlFlow(statement: ts.Statement | ts.BlockLike): boolean;
export declare type ControlFlowStatement = ts.BreakStatement | ts.ContinueStatement | ts.ReturnStatement | ts.ThrowStatement;
export interface ControlFlowEnd {
    readonly statements: ReadonlyArray<ControlFlowStatement>;
    readonly end: boolean;
}
export declare function getControlFlowEnd(statement: ts.Statement | ts.BlockLike): ControlFlowEnd;
export declare function getControlFlowEnd(statement: ts.Statement | ts.BlockLike, label?: ts.Identifier): ControlFlowEnd;
