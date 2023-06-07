import * as ts from 'typescript';
export interface NodeWrap {
    node: ts.Node;
    kind: ts.SyntaxKind;
    children: NodeWrap[];
    next?: NodeWrap;
    skip?: NodeWrap;
    parent?: NodeWrap;
}
export interface WrappedAst extends NodeWrap {
    next: NodeWrap;
    skip: undefined;
    parent: undefined;
}
export interface ConvertedAst {
    wrapped: WrappedAst;
    flat: ReadonlyArray<ts.Node>;
}
export declare function convertAst(sourceFile: ts.SourceFile): ConvertedAst;
