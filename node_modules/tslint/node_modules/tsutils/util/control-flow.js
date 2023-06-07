"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ts = require("typescript");
var node_1 = require("../typeguard/node");
function endsControlFlow(statement) {
    return getControlFlowEnd(statement).end;
}
exports.endsControlFlow = endsControlFlow;
var defaultControlFlowEnd = { statements: [], end: false };
function getControlFlowEnd(statement) {
    return node_1.isBlockLike(statement) ? handleBlock(statement) : getControlFlowEndWorker(statement);
}
exports.getControlFlowEnd = getControlFlowEnd;
function getControlFlowEndWorker(statement) {
    switch (statement.kind) {
        case ts.SyntaxKind.ReturnStatement:
        case ts.SyntaxKind.ThrowStatement:
        case ts.SyntaxKind.ContinueStatement:
        case ts.SyntaxKind.BreakStatement:
            return { statements: [statement], end: true };
        case ts.SyntaxKind.Block:
            return handleBlock(statement);
        case ts.SyntaxKind.ForStatement:
        case ts.SyntaxKind.WhileStatement:
            return handleForAndWhileStatement(statement);
        case ts.SyntaxKind.ForOfStatement:
        case ts.SyntaxKind.ForInStatement:
            return handleForInOrOfStatement(statement);
        case ts.SyntaxKind.DoStatement:
            return matchBreakOrContinue(getControlFlowEndWorker(statement.statement), node_1.isBreakOrContinueStatement);
        case ts.SyntaxKind.IfStatement:
            return handleIfStatement(statement);
        case ts.SyntaxKind.SwitchStatement:
            return matchBreakOrContinue(handleSwitchStatement(statement), node_1.isBreakStatement);
        case ts.SyntaxKind.TryStatement:
            return handleTryStatement(statement);
        case ts.SyntaxKind.LabeledStatement:
            return matchLabel(getControlFlowEndWorker(statement.statement), statement.label);
        case ts.SyntaxKind.WithStatement:
            return getControlFlowEndWorker(statement.statement);
        default:
            return defaultControlFlowEnd;
    }
}
function handleBlock(statement) {
    var _a;
    var result = { statements: [], end: false };
    for (var _i = 0, _b = statement.statements; _i < _b.length; _i++) {
        var s = _b[_i];
        var current = getControlFlowEndWorker(s);
        (_a = result.statements).push.apply(_a, current.statements);
        if (current.end) {
            result.end = true;
            break;
        }
    }
    return result;
}
function handleForInOrOfStatement(statement) {
    var end = matchBreakOrContinue(getControlFlowEndWorker(statement.statement), node_1.isBreakOrContinueStatement);
    end.end = false;
    return end;
}
function handleForAndWhileStatement(statement) {
    var constantCondition = statement.kind === ts.SyntaxKind.WhileStatement
        ? getConstantCondition(statement.expression)
        : statement.condition === undefined || getConstantCondition(statement.condition);
    if (constantCondition === false)
        return defaultControlFlowEnd;
    var end = matchBreakOrContinue(getControlFlowEndWorker(statement.statement), node_1.isBreakOrContinueStatement);
    if (constantCondition === undefined)
        end.end = false;
    return end;
}
function getConstantCondition(node) {
    switch (node.kind) {
        case ts.SyntaxKind.TrueKeyword:
            return true;
        case ts.SyntaxKind.FalseKeyword:
            return false;
        default:
            return;
    }
}
function handleIfStatement(node) {
    switch (getConstantCondition(node.expression)) {
        case true:
            return getControlFlowEndWorker(node.thenStatement);
        case false:
            return node.elseStatement === undefined
                ? defaultControlFlowEnd
                : getControlFlowEndWorker(node.elseStatement);
    }
    var then = getControlFlowEndWorker(node.thenStatement);
    if (node.elseStatement === undefined)
        return {
            statements: then.statements,
            end: false,
        };
    var elze = getControlFlowEndWorker(node.elseStatement);
    return {
        statements: then.statements.concat(elze.statements),
        end: then.end && elze.end,
    };
}
function handleSwitchStatement(node) {
    var _a;
    var hasDefault = false;
    var result = {
        statements: [],
        end: false,
    };
    for (var _i = 0, _b = node.caseBlock.clauses; _i < _b.length; _i++) {
        var clause = _b[_i];
        if (clause.kind === ts.SyntaxKind.DefaultClause)
            hasDefault = true;
        var current = handleBlock(clause);
        result.end = current.end;
        (_a = result.statements).push.apply(_a, current.statements);
    }
    if (!hasDefault)
        result.end = false;
    return result;
}
function handleTryStatement(node) {
    var finallyResult;
    if (node.finallyBlock !== undefined) {
        finallyResult = handleBlock(node.finallyBlock);
        if (finallyResult.end)
            return finallyResult;
    }
    var tryResult = handleBlock(node.tryBlock);
    if (node.catchClause === undefined)
        return { statements: finallyResult.statements.concat(tryResult.statements), end: tryResult.end };
    var catchResult = handleBlock(node.catchClause.block);
    return {
        statements: tryResult.statements
            .filter(function (s) { return s.kind !== ts.SyntaxKind.ThrowStatement; })
            .concat(catchResult.statements, finallyResult === undefined ? [] : finallyResult.statements),
        end: tryResult.end && catchResult.end,
    };
}
function matchBreakOrContinue(current, pred) {
    var result = {
        statements: [],
        end: current.end,
    };
    for (var _i = 0, _a = current.statements; _i < _a.length; _i++) {
        var statement = _a[_i];
        if (pred(statement) && statement.label === undefined) {
            result.end = false;
            continue;
        }
        result.statements.push(statement);
    }
    return result;
}
function matchLabel(current, label) {
    var result = {
        statements: [],
        end: current.end,
    };
    var labelText = label.text;
    for (var _i = 0, _a = current.statements; _i < _a.length; _i++) {
        var statement = _a[_i];
        switch (statement.kind) {
            case ts.SyntaxKind.BreakStatement:
            case ts.SyntaxKind.ContinueStatement:
                if (statement.label !== undefined && statement.label.text === labelText) {
                    result.end = false;
                    continue;
                }
        }
        result.statements.push(statement);
    }
    return result;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiY29udHJvbC1mbG93LmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiY29udHJvbC1mbG93LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7O0FBQUEsK0JBQWlDO0FBQ2pDLDBDQUE4RjtBQUU5RixTQUFnQixlQUFlLENBQUMsU0FBc0M7SUFDbEUsT0FBTyxpQkFBaUIsQ0FBQyxTQUFTLENBQUMsQ0FBQyxHQUFHLENBQUM7QUFDNUMsQ0FBQztBQUZELDBDQUVDO0FBaUJELElBQU0scUJBQXFCLEdBQW1CLEVBQUMsVUFBVSxFQUFFLEVBQUUsRUFBRSxHQUFHLEVBQUUsS0FBSyxFQUFDLENBQUM7QUFLM0UsU0FBZ0IsaUJBQWlCLENBQUMsU0FBc0M7SUFDcEUsT0FBTyxrQkFBVyxDQUFDLFNBQVMsQ0FBQyxDQUFDLENBQUMsQ0FBQyxXQUFXLENBQUMsU0FBUyxDQUFDLENBQUMsQ0FBQyxDQUFDLHVCQUF1QixDQUFDLFNBQVMsQ0FBQyxDQUFDO0FBQ2hHLENBQUM7QUFGRCw4Q0FFQztBQUVELFNBQVMsdUJBQXVCLENBQUMsU0FBdUI7SUFDcEQsUUFBUSxTQUFTLENBQUMsSUFBSSxFQUFFO1FBQ3BCLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxlQUFlLENBQUM7UUFDbkMsS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLGNBQWMsQ0FBQztRQUNsQyxLQUFLLEVBQUUsQ0FBQyxVQUFVLENBQUMsaUJBQWlCLENBQUM7UUFDckMsS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLGNBQWM7WUFDN0IsT0FBTyxFQUFDLFVBQVUsRUFBRSxDQUF1QixTQUFTLENBQUMsRUFBRSxHQUFHLEVBQUUsSUFBSSxFQUFDLENBQUM7UUFDdEUsS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLEtBQUs7WUFDcEIsT0FBTyxXQUFXLENBQVcsU0FBUyxDQUFDLENBQUM7UUFDNUMsS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLFlBQVksQ0FBQztRQUNoQyxLQUFLLEVBQUUsQ0FBQyxVQUFVLENB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