"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var ts = require("typescript");
var node_1 = require("../typeguard/node");
tslib_1.__exportStar(require("./control-flow"), exports);
function getChildOfKind(node, kind, sourceFile) {
    for (var _i = 0, _a = node.getChildren(sourceFile); _i < _a.length; _i++) {
        var child = _a[_i];
        if (child.kind === kind)
            return child;
    }
}
exports.getChildOfKind = getChildOfKind;
function isTokenKind(kind) {
    return kind >= ts.SyntaxKind.FirstToken && kind <= ts.SyntaxKind.LastToken;
}
exports.isTokenKind = isTokenKind;
function isNodeKind(kind) {
    return kind >= ts.SyntaxKind.FirstNode;
}
exports.isNodeKind = isNodeKind;
function isAssignmentKind(kind) {
    return kind >= ts.SyntaxKind.FirstAssignment && kind <= ts.SyntaxKind.LastAssignment;
}
exports.isAssignmentKind = isAssignmentKind;
function isTypeNodeKind(kind) {
    return kind >= ts.SyntaxKind.FirstTypeNode && kind <= ts.SyntaxKind.LastTypeNode;
}
exports.isTypeNodeKind = isTypeNodeKind;
function isJsDocKind(kind) {
    return kind >= ts.SyntaxKind.FirstJSDocNode && kind <= ts.SyntaxKind.LastJSDocNode;
}
exports.isJsDocKind = isJsDocKind;
function isThisParameter(parameter) {
    return parameter.name.kind === ts.SyntaxKind.Identifier && parameter.name.originalKeywordKind === ts.SyntaxKind.ThisKeyword;
}
exports.isThisParameter = isThisParameter;
function getModifier(node, kind) {
    if (node.modifiers !== undefined)
        for (var _i = 0, _a = node.modifiers; _i < _a.length; _i++) {
            var modifier = _a[_i];
            if (modifier.kind === kind)
                return modifier;
        }
}
exports.getModifier = getModifier;
function hasModifier(modifiers) {
    var kinds = [];
    for (var _i = 1; _i < arguments.length; _i++) {
        kinds[_i - 1] = arguments[_i];
    }
    if (modifiers === undefined)
        return false;
    for (var _a = 0, modifiers_1 = modifiers; _a < modifiers_1.length; _a++) {
        var modifier = modifiers_1[_a];
        if (kinds.indexOf(modifier.kind) !== -1)
            return true;
    }
    return false;
}
exports.hasModifier = hasModifier;
function isParameterProperty(node) {
    return hasModifier(node.modifiers, ts.SyntaxKind.PublicKeyword, ts.SyntaxKind.ProtectedKeyword, ts.SyntaxKind.PrivateKeyword, ts.SyntaxKind.ReadonlyKeyword);
}
exports.isParameterProperty = isParameterProperty;
function hasAccessModifier(node) {
    return hasModifier(node.modifiers, ts.SyntaxKind.PublicKeyword, ts.SyntaxKind.ProtectedKeyword, ts.SyntaxKind.PrivateKeyword);
}
exports.hasAccessModifier = hasAccessModifier;
function isFlagSet(obj, flag) {
    return (obj.flags & flag) !== 0;
}
exports.isNodeFlagSet = isFlagSet;
exports.isTypeFlagSet = isFlagSet;
exports.isSymbolFlagSet = isFlagSet;
function isObjectFlagSet(objectType, flag) {
    return (objectType.objectFlags & flag) !== 0;
}
exports.isObjectFlagSet = isObjectFlagSet;
function isModifierFlagSet(node, flag) {
    return (ts.getCombinedModifierFlags(node) & flag) !== 0;
}
exports.isModifierFlagSet = isModifierFlagSet;
function isModfierFlagSet(node, flag) {
    return isModifierFlagSet(node, flag);
}
exports.isModfierFlagSet = isModfierFlagSet;
function getPreviousStatement(statement) {
    var parent = statement.parent;
    if (node_1.isBlockLike(parent)) {
        var index = parent.statements.indexOf(statement);
        if (index > 0)
            return parent.statements[index - 1];
    }
}
exports.getPreviousStatement = getPreviousStatement;
function getNextStatement(statement) {
    var parent = statement.parent;
    if (node_1.isBlockLike(parent)) {
        var index = parent.statements.indexOf(statement);
        if (index < parent.statements.length)
            return parent.statements[index + 1];
    }
}
exports.getNextStatement = getNextStatement;
function getPreviousToken(node, sourceFile) {
    var parent = node.parent;
    while (parent !== undefined && parent.pos === node.pos)
        parent = parent.parent;
    if (parent === undefined)
        return;
    outer: while (true) {
        var children = parent.getChildren(sourceFile);
        for (var i = children.length - 1; i >= 0; --i) {
            var child = children[i];
            if (child.pos < node.pos && child.kind !== ts.SyntaxKind.JSDocComment) {
                if (isTokenKind(child.kind))
                    return child;
                parent = child;
                continue outer;
            }
        }
        return;
    }
}
exports.getPreviousToken = getPreviousToken;
function getNextToken(node, sourceFile) {
    if (sourceFile === void 0) { sourceFile = node.getSourceFile(); }
    if (node.kind === ts.SyntaxKind.SourceFile || node.kind === ts.SyntaxKind.EndOfFileToken)
        return;
    var end = node.end;
    node = node.parent;
    while (node.end === end) {
        if (node.parent === undefined)
            return node.endOfFileToken;
        node = node.parent;
    }
    return getTokenAtPositionWorker(node, end, sourceFile);
}
exports.getNextToken = getNextToken;
function getTokenAtPosition(parent, pos, sourceFile) {
    if (pos < parent.pos || pos >= parent.end)
        return;
    if (isTokenKind(parent.kind))
        return parent;
    if (sourceFile === undefined)
        sourceFile = parent.getSourceFile();
    return getTokenAtPositionWorker(parent, pos, sourceFile);
}
exports.getTokenAtPosition = getTokenAtPosition;
function getTokenAtPositionWorker(node, pos, sourceFile) {
    outer: while (true) {
        for (var _i = 0, _a = node.getChildren(sourceFile); _i < _a.length; _i++) {
            var child = _a[_i];
            if (child.end > pos && child.kind !== ts.SyntaxKind.JSDocComment) {
                if (isTokenKind(child.kind))
                    return child;
                node = child;
                continue outer;
            }
        }
        return;
    }
}
function getCommentAtPosition(sourceFile, pos, parent) {
    if (parent === void 0) { parent = sourceFile; }
    var token = getTokenAtPosition(parent, pos, sourceFile);
    if (token === undefined || token.kind === ts.SyntaxKind.JsxText || pos >= token.end - (ts.tokenToString(token.kind) || '').length)
        return;
    var startPos = token.pos === 0
        ? (ts.getShebang(sourceFile.text) || '').length
        : token.pos;
    return startPos !== 0 && ts.forEachTrailingCommentRange(sourceFile.text, startPos, commentAtPositionCallback, pos) ||
        ts.forEachLeadingCommentRange(sourceFile.text, startPos, commentAtPositionCallback, pos);
}
exports.getCommentAtPosition = getCommentAtPosition;
function commentAtPositionCallback(pos, end, kind, _nl, at) {
    return at >= pos && at < end ? { pos: pos, end: end, kind: kind } : undefined;
}
function isPositionInComment(sourceFile, pos, parent) {
    return getCommentAtPosition(sourceFile, pos, parent) !== undefined;
}
exports.isPositionInComment = isPositionInComment;
function getWrappedNodeAtPosition(wrap, pos) {
    if (wrap.node.pos > pos || wrap.node.end <= pos)
        return;
    outer: while (true) {
        for (var _i = 0, _a = wrap.children; _i < _a.length; _i++) {
            var child = _a[_i];
            if (child.node.pos > pos)
                return wrap;
            if (child.node.end > pos) {
                wrap = child;
                continue outer;
            }
        }
        return wrap;
    }
}
exports.getWrappedNodeAtPosition = getWrappedNodeAtPosition;
function getPropertyName(propertyName) {
    if (propertyName.kind === ts.SyntaxKind.ComputedPropertyName) {
        if (!node_1.isLiteralExpression(propertyName.expression))
            return;
        return propertyName.expression.text;
    }
    return propertyName.kind === ts.SyntaxKind.Identifier ? getIdentifierText(propertyName) : propertyName.text;
}
exports.getPropertyName = getPropertyName;
function forEachDestructuringIdentifier(pattern, fn) {
    for (var _i = 0, _a = pattern.elements; _i < _a.length; _i++) {
        var element = _a[_i];
        if (element.kind !== ts.SyntaxKind.BindingElement)
            continue;
        var result = void 0;
        if (element.name.kind === ts.SyntaxKind.Identifier) {
            result = fn(element);
        }
        else {
            result = forEachDestructuringIdentifier(element.name, fn);
        }
        if (result)
            return result;
    }
}
exports.forEachDestructuringIdentifier = forEachDestructuringIdentifier;
function forEachDeclaredVariable(declarationList, cb) {
    for (var _i = 0, _a = declarationList.declarations; _i < _a.length; _i++) {
        var declaration = _a[_i];
        var result = void 0;
        if (declaration.name.kind === ts.SyntaxKind.Identifier) {
            result = cb(declaration);
        }
        else {
            result = forEachDestructuringIdentifier(declaration.name, cb);
        }
        if (result)
            return result;
    }
}
exports.forEachDeclaredVariable = forEachDeclaredVariable;
var VariableDeclarationKind;
(function (VariableDeclarationKind) {
    VariableDeclarationKind[VariableDeclarationKind["Var"] = 0] = "Var";
    VariableDeclarationKind[VariableDeclarationKind["Let"] = 1] = "Let";
    VariableDeclarationKind[VariableDeclarationKind["Const"] = 2] = "Const";
})(VariableDeclarationKind = exports.VariableDeclarationKind || (exports.VariableDeclarationKind = {}));
function getVariableDeclarationKind(declarationList) {
    if (declarationList.flags & ts.NodeFlags.Let)
        return 1;
    if (declarationList.flags & ts.NodeFlags.Const)
        return 2;
    return 0;
}
exports.getVariableDeclarationKind = getVariableDeclarationKind;
function isBlockScopedVariableDeclarationList(declarationList) {
    return (declarationList.flags & ts.NodeFlags.BlockScoped) !== 0;
}
exports.isBlockScopedVariableDeclarationList = isBlockScopedVariableDeclarationList;
function isBlockScopedVariableDeclaration(declaration) {
    var parent = declaration.parent;
    return parent.kind === ts.SyntaxKind.CatchClause ||
        isBlockScopedVariableDeclarationList(parent);
}
exports.isBlockScopedVariableDeclaration = isBlockScopedVariableDeclaration;
var ScopeBoundary;
(function (ScopeBoundary) {
    ScopeBoundary[ScopeBoundary["None"] = 0] = "None";
    ScopeBoundary[ScopeBoundary["Function"] = 1] = "Function";
    ScopeBoundary[ScopeBoundary["Block"] = 2] = "Block";
})(ScopeBoundary = exports.ScopeBoundary || (exports.ScopeBoundary = {}));
function isScopeBoundary(node) {
    if (isFunctionScopeBoundary(node))
        return 1;
    if (isBlockScopeBoundary(node))
        return 2;
    return 0;
}
exports.isScopeBoundary = isScopeBoundary;
function isFunctionScopeBoundary(node) {
    switch (node.kind) {
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.ArrowFunction:
        case ts.SyntaxKind.Constructor:
        case ts.SyntaxKind.ModuleDeclaration:
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.EnumDeclaration:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.SetAccessor:
        case ts.SyntaxKind.InterfaceDeclaration:
        case ts.SyntaxKind.TypeAliasDeclaration:
        case ts.SyntaxKind.MethodSignature:
        case ts.SyntaxKind.CallSignature:
        case ts.SyntaxKind.ConstructSignature:
        case ts.SyntaxKind.ConstructorType:
        case ts.SyntaxKind.FunctionType:
        case ts.SyntaxKind.MappedType:
        case ts.SyntaxKind.ConditionalType:
            return true;
        case ts.SyntaxKind.SourceFile:
            return ts.isExternalModule(node);
        default:
            return false;
    }
}
exports.isFunctionScopeBoundary = isFunctionScopeBoundary;
function isBlockScopeBoundary(node) {
    switch (node.kind) {
        case ts.SyntaxKind.Block:
            var parent = node.parent;
            return parent.kind !== ts.SyntaxKind.CatchClause &&
                (parent.kind === ts.SyntaxKind.SourceFile ||
                    !isFunctionScopeBoundary(parent));
        case ts.SyntaxKind.ForStatement:
        case ts.SyntaxKind.ForInStatement:
        case ts.SyntaxKind.ForOfStatement:
        case ts.SyntaxKind.CaseBlock:
        case ts.SyntaxKind.CatchClause:
            return true;
        default:
            return false;
    }
}
exports.isBlockScopeBoundary = isBlockScopeBoundary;
function hasOwnThisReference(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.FunctionExpression:
            return true;
        case ts.SyntaxKind.FunctionDeclaration:
            return node.body !== undefined;
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.SetAccessor:
            return node.parent.kind === ts.SyntaxKind.ObjectLiteralExpression;
        default:
            return false;
    }
}
exports.hasOwnThisReference = hasOwnThisReference;
function isFunctionWithBody(node) {
    switch (node.kind) {
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.SetAccessor:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.Constructor:
            return node.body !== undefined;
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.ArrowFunction:
            return true;
        default:
            return false;
    }
}
exports.isFunctionWithBody = isFunctionWithBody;
function forEachToken(node, cb, sourceFile) {
    if (sourceFile === void 0) { sourceFile = node.getSourceFile(); }
    return (function iterate(child) {
        if (isTokenKind(child.kind))
            return cb(child);
        if (child.kind !== ts.SyntaxKind.JSDocComment)
            return child.getChildren(sourceFile).forEach(iterate);
    })(node);
}
exports.forEachToken = forEachToken;
function forEachTokenWithTrivia(node, cb, sourceFile) {
    if (sourceFile === void 0) { sourceFile = node.getSourceFile(); }
    var fullText = sourceFile.text;
    var scanner = ts.createScanner(sourceFile.languageVersion, false, sourceFile.languageVariant, fullText);
    return forEachToken(node, function (token) {
        var tokenStart = token.kind === ts.SyntaxKind.JsxText || token.pos === token.end ? token.pos : token.getStart(sourceFile);
        if (tokenStart !== token.pos) {
            scanner.setTextPos(token.pos);
            var kind = scanner.scan();
            var pos = scanner.getTokenPos();
            while (pos < tokenStart) {
                var textPos = scanner.getTextPos();
                cb(fullText, kind, { pos: pos, end: textPos }, token.parent);
                if (textPos === tokenStart)
                    break;
                kind = scanner.scan();
                pos = scanner.getTokenPos();
            }
        }
        return cb(fullText, token.kind, { end: token.end, pos: tokenStart }, token.parent);
    }, sourceFile);
}
exports.forEachTokenWithTrivia = forEachTokenWithTrivia;
function forEachComment(node, cb, sourceFile) {
    if (sourceFile === void 0) { sourceFile = node.getSourceFile(); }
    var fullText = sourceFile.text;
    var notJsx = sourceFile.languageVariant !== ts.LanguageVariant.JSX;
    return forEachToken(node, function (token) {
        if (token.pos === token.end)
            return;
        if (token.kind !== ts.SyntaxKind.JsxText)
            ts.forEachLeadingCommentRange(fullText, token.pos === 0 ? (ts.getShebang(fullText) || '').length : token.pos, commentCallback);
        if (notJsx || canHaveTrailingTrivia(token))
            return ts.forEachTrailingCommentRange(fullText, token.end, commentCallback);
    }, sourceFile);
    function commentCallback(pos, end, kind) {
        cb(fullText, { pos: pos, end: end, kind: kind });
    }
}
exports.forEachComment = forEachComment;
function canHaveTrailingTrivia(token) {
    switch (token.kind) {
        case ts.SyntaxKind.CloseBraceToken:
            return token.parent.kind !== ts.SyntaxKind.JsxExpression || !isJsxElementOrFragment(token.parent.parent);
        case ts.SyntaxKind.GreaterThanToken:
            switch (token.parent.kind) {
                case ts.SyntaxKind.JsxOpeningElement:
                    return token.end !== token.parent.end;
                case ts.SyntaxKind.JsxOpeningFragment:
                    return false;
                case ts.SyntaxKind.JsxSelfClosingElement:
                    return token.end !== token.parent.end ||
                        !isJsxElementOrFragment(token.parent.parent);
                case ts.SyntaxKind.JsxClosingElement:
                case ts.SyntaxKind.JsxClosingFragment:
                    return !isJsxElementOrFragment(token.parent.parent.parent);
            }
    }
    return true;
}
function isJsxElementOrFragment(node) {
    return node.kind === ts.SyntaxKind.JsxElement || node.kind === ts.SyntaxKind.JsxFragment;
}
function getLineRanges(sourceFile) {
    var lineStarts = sourceFile.getLineStarts();
    var result = [];
    var length = lineStarts.length;
    var sourceText = sourceFile.text;
    var pos = 0;
    for (var i = 1; i < length; ++i) {
        var end = lineStarts[i];
        var lineEnd = end;
        for (; lineEnd > pos; --lineEnd)
            if (!ts.isLineBreak(sourceText.charCodeAt(lineEnd - 1)))
                break;
        result.push({
            pos: pos,
            end: end,
            contentLength: lineEnd - pos,
        });
        pos = end;
    }
    result.push({
        pos: pos,
        end: sourceFile.end,
        contentLength: sourceFile.end - pos,
    });
    return result;
}
exports.getLineRanges = getLineRanges;
function getLineBreakStyle(sourceFile) {
    var lineStarts = sourceFile.getLineStarts();
    return lineStarts.length === 1 || lineStarts[1] < 2 || sourceFile.text[lineStarts[1] - 2] !== '\r'
        ? '\n'
        : '\r\n';
}
exports.getLineBreakStyle = getLineBreakStyle;
var cachedScanner;
function scanToken(text) {
    if (cachedScanner === undefined)
        cachedScanner = ts.createScanner(ts.ScriptTarget.Latest, false);
    cachedScanner.setText(text);
    cachedScanner.scan();
    return cachedScanner;
}
function isValidIdentifier(text) {
    var scan = scanToken(text);
    return scan.isIdentifier() && scan.getTextPos() === text.length && scan.getTokenPos() === 0;
}
exports.isValidIdentifier = isValidIdentifier;
function isValidPropertyAccess(text) {
    if (!ts.isIdentifierStart(text.charCodeAt(0), ts.ScriptTarget.Latest))
        return false;
    for (var i = 1; i < text.length; ++i)
        if (!ts.isIdentifierPart(text.charCodeAt(i), ts.ScriptTarget.Latest))
            return false;
    return true;
}
exports.isValidPropertyAccess = isValidPropertyAccess;
function isValidPropertyName(text) {
    if (isValidPropertyAccess(text))
        return true;
    var scan = scanToken(text);
    return scan.getTextPos() === text.length &&
        scan.getToken() === ts.SyntaxKind.NumericLiteral && scan.getTokenValue() === text;
}
exports.isValidPropertyName = isValidPropertyName;
function isValidNumericLiteral(text) {
    var scan = scanToken(text);
    return scan.getToken() === ts.SyntaxKind.NumericLiteral && scan.getTextPos() === text.length && scan.getTokenPos() === 0;
}
exports.isValidNumericLiteral = isValidNumericLiteral;
function isSameLine(sourceFile, pos1, pos2) {
    return ts.getLineAndCharacterOfPosition(sourceFile, pos1).line === ts.getLineAndCharacterOfPosition(sourceFile, pos2).line;
}
exports.isSameLine = isSameLine;
var SideEffectOptions;
(function (SideEffectOptions) {
    SideEffectOptions[SideEffectOptions["None"] = 0] = "None";
    SideEffectOptions[SideEffectOptions["TaggedTemplate"] = 1] = "TaggedTemplate";
    SideEffectOptions[SideEffectOptions["Constructor"] = 2] = "Constructor";
    SideEffectOptions[SideEffectOptions["JsxElement"] = 4] = "JsxElement";
})(SideEffectOptions = exports.SideEffectOptions || (exports.SideEffectOptions = {}));
function hasSideEffects(node, options) {
    switch (node.kind) {
        case ts.SyntaxKind.CallExpression:
        case ts.SyntaxKind.PostfixUnaryExpression:
        case ts.SyntaxKind.AwaitExpression:
        case ts.SyntaxKind.YieldExpression:
        case ts.SyntaxKind.DeleteExpression:
            return true;
        case ts.SyntaxKind.TypeAssertionExpression:
        case ts.SyntaxKind.AsExpression:
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.NonNullExpression:
        case ts.SyntaxKind.VoidExpression:
        case ts.SyntaxKind.TypeOfExpression:
        case ts.SyntaxKind.PropertyAccessExpression:
        case ts.SyntaxKind.SpreadElement:
        case ts.SyntaxKind.PartiallyEmittedExpression:
            return hasSideEffects(node.expression, options);
        case ts.SyntaxKind.BinaryExpression:
            return isAssignmentKind(node.operatorToken.kind) ||
                hasSideEffects(node.left, options) ||
                hasSideEffects(node.right, options);
        case ts.SyntaxKind.PrefixUnaryExpression:
            switch (node.operator) {
                case ts.SyntaxKind.PlusPlusToken:
                case ts.SyntaxKind.MinusMinusToken:
                    return true;
                default:
                    return hasSideEffects(node.operand, options);
            }
        case ts.SyntaxKind.ElementAccessExpression:
            return hasSideEffects(node.expression, options) ||
                node.argumentExpression !== undefined &&
                    hasSideEffects(node.argumentExpression, options);
        case ts.SyntaxKind.ConditionalExpression:
            return hasSideEffects(node.condition, options) ||
                hasSideEffects(node.whenTrue, options) ||
                hasSideEffects(node.whenFalse, options);
        case ts.SyntaxKind.NewExpression:
            if (options & 2 || hasSideEffects(node.expression, options))
                return true;
            if (node.arguments !== undefined)
                for (var _i = 0, _a = node.arguments; _i < _a.length; _i++) {
                    var child = _a[_i];
                    if (hasSideEffects(child, options))
                        return true;
                }
            return false;
        case ts.SyntaxKind.TaggedTemplateExpression:
            if (options & 1 || hasSideEffects(node.tag, options))
                return true;
            if (node.template.kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral)
                return false;
            node = node.template;
        case ts.SyntaxKind.TemplateExpression:
            for (var _b = 0, _c = node.templateSpans; _b < _c.length; _b++) {
                var child = _c[_b];
                if (hasSideEffects(child.expression, options))
                    return true;
            }
            return false;
        case ts.SyntaxKind.ClassExpression:
            return classExpressionHasSideEffects(node, options);
        case ts.SyntaxKind.ArrayLiteralExpression:
            for (var _d = 0, _e = node.elements; _d < _e.length; _d++) {
                var child = _e[_d];
                if (hasSideEffects(child, options))
                    return true;
            }
            return false;
        case ts.SyntaxKind.ObjectLiteralExpression:
            for (var _f = 0, _g = node.properties; _f < _g.length; _f++) {
                var child = _g[_f];
                if (child.name !== undefined && child.name.kind === ts.SyntaxKind.ComputedPropertyName &&
                    hasSideEffects(child.name.expression, options))
                    return true;
                switch (child.kind) {
                    case ts.SyntaxKind.PropertyAssignment:
                        if (hasSideEffects(child.initializer, options))
                            return true;
                        break;
                    case ts.SyntaxKind.SpreadAssignment:
                        if (hasSideEffects(child.expression, options))
                            return true;
                }
            }
            return false;
        case ts.SyntaxKind.JsxExpression:
            return node.expression !== undefined && hasSideEffects(node.expression, options);
        case ts.SyntaxKind.JsxElement:
        case ts.SyntaxKind.JsxFragment:
            for (var _h = 0, _j = node.children; _h < _j.length; _h++) {
                var child = _j[_h];
                if (child.kind !== ts.SyntaxKind.JsxText && hasSideEffects(child, options))
                    return true;
            }
            if (node.kind === ts.SyntaxKind.JsxFragment)
                return false;
            node = node.openingElement;
        case ts.SyntaxKind.JsxSelfClosingElement:
        case ts.SyntaxKind.JsxOpeningElement:
            if (options & 4)
                return true;
            for (var _k = 0, _l = getJsxAttributes(node); _k < _l.length; _k++) {
                var child = _l[_k];
                if (child.kind === ts.SyntaxKind.JsxSpreadAttribute) {
                    if (hasSideEffects(child.expression, options))
                        return true;
                }
                else if (child.initializer !== undefined && hasSideEffects(child.initializer, options)) {
                    return true;
                }
            }
            return false;
        case ts.SyntaxKind.CommaListExpression:
            for (var _m = 0, _o = node.elements; _m < _o.length; _m++) {
                var child = _o[_m];
                if (hasSideEffects(child, options))
                    return true;
            }
            return false;
        default:
            return false;
    }
}
exports.hasSideEffects = hasSideEffects;
function getJsxAttributes(openElement) {
    var attributes = openElement.attributes;
    return Array.isArray(attributes) ? attributes : attributes.properties;
}
function classExpressionHasSideEffects(node, options) {
    if (node.heritageClauses !== undefined && node.heritageClauses[0].token === ts.SyntaxKind.ExtendsKeyword)
        for (var _i = 0, _a = node.heritageClauses[0].types; _i < _a.length; _i++) {
            var base = _a[_i];
            if (hasSideEffects(base.expression, options))
                return true;
        }
    for (var _b = 0, _c = node.members; _b < _c.length; _b++) {
        var child = _c[_b];
        if (child.name !== undefined && child.name.kind === ts.SyntaxKind.ComputedPropertyName &&
            hasSideEffects(child.name.expression, options) ||
            node_1.isPropertyDeclaration(child) && child.initializer !== undefined &&
                hasSideEffects(child.initializer, options))
            return true;
    }
    return false;
}
function getDeclarationOfBindingElement(node) {
    var parent = node.parent.parent;
    while (parent.kind === ts.SyntaxKind.BindingElement)
        parent = parent.parent.parent;
    return parent;
}
exports.getDeclarationOfBindingElement = getDeclarationOfBindingElement;
function isExpressionValueUsed(node) {
    while (true) {
        var parent = node.parent;
        switch (parent.kind) {
            case ts.SyntaxKind.CallExpression:
            case ts.SyntaxKind.NewExpression:
            case ts.SyntaxKind.ElementAccessExpression:
            case ts.SyntaxKind.WhileStatement:
            case ts.SyntaxKind.DoStatement:
            case ts.SyntaxKind.WithStatement:
            case ts.SyntaxKind.ThrowStatement:
            case ts.SyntaxKind.ReturnStatement:
            case ts.SyntaxKind.JsxExpression:
            case ts.SyntaxKind.JsxSpreadAttribute:
            case ts.SyntaxKind.JsxElement:
            case ts.SyntaxKind.JsxFragment:
            case ts.SyntaxKind.JsxSelfClosingElement:
            case ts.SyntaxKind.ComputedPropertyName:
            case ts.SyntaxKind.ArrowFunction:
            case ts.SyntaxKind.ExportSpecifier:
            case ts.SyntaxKind.ExportAssignment:
            case ts.SyntaxKind.ImportDeclaration:
            case ts.SyntaxKind.ExternalModuleReference:
            case ts.SyntaxKind.Decorator:
            case ts.SyntaxKind.TaggedTemplateExpression:
            case ts.SyntaxKind.TemplateSpan:
            case ts.SyntaxKind.ExpressionWithTypeArguments:
            case ts.SyntaxKind.TypeOfExpression:
            case ts.SyntaxKind.AwaitExpression:
            case ts.SyntaxKind.YieldExpression:
            case ts.SyntaxKind.LiteralType:
            case ts.SyntaxKind.JsxAttributes:
            case ts.SyntaxKind.JsxOpeningElement:
            case ts.SyntaxKind.JsxClosingElement:
            case ts.SyntaxKind.IfStatement:
            case ts.SyntaxKind.CaseClause:
            case ts.SyntaxKind.SwitchStatement:
                return true;
            case ts.SyntaxKind.PropertyAccessExpression:
                return parent.expression === node;
            case ts.SyntaxKind.QualifiedName:
                return parent.left === node;
            case ts.SyntaxKind.ShorthandPropertyAssignment:
                return parent.objectAssignmentInitializer === node ||
                    !isInDestructuringAssignment(parent);
            case ts.SyntaxKind.PropertyAssignment:
                return parent.initializer === node && !isInDestructuringAssignment(parent);
            case ts.SyntaxKind.SpreadAssignment:
            case ts.SyntaxKind.SpreadElement:
            case ts.SyntaxKind.ArrayLiteralExpression:
                return !isInDestructuringAssignment(parent);
            case ts.SyntaxKind.ParenthesizedExpression:
            case ts.SyntaxKind.AsExpression:
            case ts.SyntaxKind.TypeAssertionExpression:
            case ts.SyntaxKind.PostfixUnaryExpression:
            case ts.SyntaxKind.PrefixUnaryExpression:
            case ts.SyntaxKind.NonNullExpression:
                node = parent;
                break;
            case ts.SyntaxKind.ForStatement:
                return parent.condition === node;
            case ts.SyntaxKind.ForInStatement:
            case ts.SyntaxKind.ForOfStatement:
                return parent.expression === node;
            case ts.SyntaxKind.ConditionalExpression:
                if (parent.condition === node)
                    return true;
                node = parent;
                break;
            case ts.SyntaxKind.PropertyDeclaration:
            case ts.SyntaxKind.BindingElement:
            case ts.SyntaxKind.VariableDeclaration:
            case ts.SyntaxKind.Parameter:
            case ts.SyntaxKind.EnumMember:
                return parent.initializer === node;
            case ts.SyntaxKind.ImportEqualsDeclaration:
                return parent.moduleReference === node;
            case ts.SyntaxKind.CommaListExpression:
                if (parent.elements[parent.elements.length - 1] !== node)
                    return false;
                node = parent;
                break;
            case ts.SyntaxKind.BinaryExpression:
                if (parent.right === node) {
                    if (parent.operatorToken.kind === ts.SyntaxKind.CommaToken) {
                        node = parent;
                        break;
                    }
                    return true;
                }
                switch (parent.operatorToken.kind) {
                    case ts.SyntaxKind.CommaToken:
                    case ts.SyntaxKind.EqualsToken:
                        return false;
                    case ts.SyntaxKind.EqualsEqualsEqualsToken:
                    case ts.SyntaxKind.EqualsEqualsToken:
                    case ts.SyntaxKind.ExclamationEqualsEqualsToken:
                    case ts.SyntaxKind.ExclamationEqualsToken:
                    case ts.SyntaxKind.InstanceOfKeyword:
                    case ts.SyntaxKind.PlusToken:
                    case ts.SyntaxKind.MinusToken:
                    case ts.SyntaxKind.AsteriskToken:
                    case ts.SyntaxKind.SlashToken:
                    case ts.SyntaxKind.PercentToken:
                    case ts.SyntaxKind.AsteriskAsteriskToken:
                    case ts.SyntaxKind.GreaterThanToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanToken:
                    case ts.SyntaxKind.GreaterThanGreaterThanGreaterThanToken:
                    case ts.SyntaxKind.GreaterThanEqualsToken:
                    case ts.SyntaxKind.LessThanToken:
                    case ts.SyntaxKind.LessThanLessThanToken:
                    case ts.SyntaxKind.LessThanEqualsToken:
                    case ts.SyntaxKind.AmpersandToken:
                    case ts.SyntaxKind.BarToken:
                    case ts.SyntaxKind.CaretToken:
                    case ts.SyntaxKind.BarBarToken:
                    case ts.SyntaxKind.AmpersandAmpersandToken:
                    case ts.SyntaxKind.InKeyword:
                        return true;
                    default:
                        node = parent;
                }
                break;
            default:
                return false;
        }
    }
}
exports.isExpressionValueUsed = isExpressionValueUsed;
function isInDestructuringAssignment(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ShorthandPropertyAssignment:
            if (node.objectAssignmentInitializer !== undefined)
                return true;
        case ts.SyntaxKind.PropertyAssignment:
        case ts.SyntaxKind.SpreadAssignment:
            node = node.parent;
            break;
        case ts.SyntaxKind.SpreadElement:
            if (node.parent.kind !== ts.SyntaxKind.ArrayLiteralExpression)
                return false;
            node = node.parent;
    }
    while (true) {
        switch (node.parent.kind) {
            case ts.SyntaxKind.BinaryExpression:
                return node.parent.left === node &&
                    node.parent.operatorToken.kind === ts.SyntaxKind.EqualsToken;
            case ts.SyntaxKind.ForOfStatement:
                return node.parent.initializer === node;
            case ts.SyntaxKind.ArrayLiteralExpression:
            case ts.SyntaxKind.ObjectLiteralExpression:
                node = node.parent;
                break;
            case ts.SyntaxKind.SpreadAssignment:
            case ts.SyntaxKind.PropertyAssignment:
                node = node.parent.parent;
                break;
            case ts.SyntaxKind.SpreadElement:
                if (node.parent.parent.kind !== ts.SyntaxKind.ArrayLiteralExpression)
                    return false;
                node = node.parent.parent;
                break;
            default:
                return false;
        }
    }
}
function isReassignmentTarget(node) {
    var parent = node.parent;
    switch (parent.kind) {
        case ts.SyntaxKind.PostfixUnaryExpression:
        case ts.SyntaxKind.DeleteExpression:
            return true;
        case ts.SyntaxKind.PrefixUnaryExpression:
            return parent.operator === ts.SyntaxKind.PlusPlusToken ||
                parent.operator === ts.SyntaxKind.MinusMinusToken;
        case ts.SyntaxKind.BinaryExpression:
            return parent.left === node &&
                isAssignmentKind(parent.operatorToken.kind);
        case ts.SyntaxKind.ShorthandPropertyAssignment:
            return parent.name === node &&
                isInDestructuringAssignment(parent);
        case ts.SyntaxKind.PropertyAssignment:
            return parent.initializer === node &&
                isInDestructuringAssignment(parent);
        case ts.SyntaxKind.ArrayLiteralExpression:
        case ts.SyntaxKind.SpreadElement:
        case ts.SyntaxKind.SpreadAssignment:
            return isInDestructuringAssignment(parent);
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.NonNullExpression:
        case ts.SyntaxKind.TypeAssertionExpression:
        case ts.SyntaxKind.AsExpression:
            return isReassignmentTarget(parent);
        case ts.SyntaxKind.ForOfStatement:
        case ts.SyntaxKind.ForInStatement:
            return parent.initializer === node;
    }
    return false;
}
exports.isReassignmentTarget = isReassignmentTarget;
function getIdentifierText(node) {
    return ts.unescapeIdentifier ? ts.unescapeIdentifier(node.text) : node.text;
}
exports.getIdentifierText = getIdentifierText;
function canHaveJsDoc(node) {
    var kind = node.kind;
    switch (kind) {
        case ts.SyntaxKind.Parameter:
        case ts.SyntaxKind.CallSignature:
        case ts.SyntaxKind.ConstructSignature:
        case ts.SyntaxKind.MethodSignature:
        case ts.SyntaxKind.PropertySignature:
        case ts.SyntaxKind.ArrowFunction:
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.SpreadAssignment:
        case ts.SyntaxKind.ShorthandPropertyAssignment:
        case ts.SyntaxKind.PropertyAssignment:
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.LabeledStatement:
        case ts.SyntaxKind.ExpressionStatement:
        case ts.SyntaxKind.VariableStatement:
        case ts.SyntaxKind.Constructor:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.PropertyDeclaration:
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.SetAccessor:
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.InterfaceDeclaration:
        case ts.SyntaxKind.TypeAliasDeclaration:
        case ts.SyntaxKind.EnumMember:
        case ts.SyntaxKind.EnumDeclaration:
        case ts.SyntaxKind.ModuleDeclaration:
        case ts.SyntaxKind.ImportEqualsDeclaration:
        case ts.SyntaxKind.IndexSignature:
        case ts.SyntaxKind.FunctionType:
        case ts.SyntaxKind.ConstructorType:
        case ts.SyntaxKind.JSDocFunctionType:
        case ts.SyntaxKind.EndOfFileToken:
            return true;
        default:
            return false;
    }
}
exports.canHaveJsDoc = canHaveJsDoc;
function getJsDoc(node, sourceFile) {
    if (node.kind === ts.SyntaxKind.EndOfFileToken)
        return parseJsDocWorker(node, sourceFile || node.parent);
    var result = [];
    for (var _i = 0, _a = node.getChildren(sourceFile); _i < _a.length; _i++) {
        var child = _a[_i];
        if (!node_1.isJsDoc(child))
            break;
        result.push(child);
    }
    return result;
}
exports.getJsDoc = getJsDoc;
function parseJsDocOfNode(node, considerTrailingComments, sourceFile) {
    if (sourceFile === void 0) { sourceFile = node.getSourceFile(); }
    if (canHaveJsDoc(node) && node.kind !== ts.SyntaxKind.EndOfFileToken) {
        var result = getJsDoc(node, sourceFile);
        if (result.length !== 0 || !considerTrailingComments)
            return result;
    }
    return parseJsDocWorker(node, sourceFile, considerTrailingComments);
}
exports.parseJsDocOfNode = parseJsDocOfNode;
function parseJsDocWorker(node, sourceFile, considerTrailingComments) {
    var nodeStart = node.getStart(sourceFile);
    var start = ts[considerTrailingComments && isSameLine(sourceFile, node.pos, nodeStart)
        ? 'forEachTrailingCommentRange'
        : 'forEachLeadingCommentRange'](sourceFile.text, node.pos, function (pos, _end, kind) { return kind === ts.SyntaxKind.MultiLineCommentTrivia && sourceFile.text[pos + 2] === '*' ? { pos: pos } : undefined; });
    if (start === undefined)
        return [];
    var startPos = start.pos;
    var text = sourceFile.text.slice(startPos, nodeStart);
    var newSourceFile = ts.createSourceFile('jsdoc.ts', text + "var a;", sourceFile.languageVersion);
    var result = getJsDoc(newSourceFile.statements[0], newSourceFile);
    for (var _i = 0, result_1 = result; _i < result_1.length; _i++) {
        var doc = result_1[_i];
        updateNode(doc, node);
    }
    return result;
    function updateNode(n, parent) {
        n.pos += startPos;
        n.end += startPos;
        n.parent = parent;
        return ts.forEachChild(n, function (child) { return updateNode(child, n); }, function (children) {
            children.pos += startPos;
            children.end += startPos;
            for (var _i = 0, children_1 = children; _i < children_1.length; _i++) {
                var child = children_1[_i];
                updateNode(child, n);
            }
        });
    }
}
var ImportKind;
(function (ImportKind) {
    ImportKind[ImportKind["ImportDeclaration"] = 1] = "ImportDeclaration";
    ImportKind[ImportKind["ImportEquals"] = 2] = "ImportEquals";
    ImportKind[ImportKind["ExportFrom"] = 4] = "ExportFrom";
    ImportKind[ImportKind["DynamicImport"] = 8] = "DynamicImport";
    ImportKind[ImportKind["Require"] = 16] = "Require";
    ImportKind[ImportKind["ImportType"] = 32] = "ImportType";
    ImportKind[ImportKind["All"] = 63] = "All";
    ImportKind[ImportKind["AllImports"] = 59] = "AllImports";
    ImportKind[ImportKind["AllStaticImports"] = 3] = "AllStaticImports";
    ImportKind[ImportKind["AllImportExpressions"] = 24] = "AllImportExpressions";
    ImportKind[ImportKind["AllRequireLike"] = 18] = "AllRequireLike";
    ImportKind[ImportKind["AllNestedImports"] = 56] = "AllNestedImports";
})(ImportKind = exports.ImportKind || (exports.ImportKind = {}));
var ImportOptions;
(function (ImportOptions) {
    ImportOptions[ImportOptions["ImportDeclaration"] = 1] = "ImportDeclaration";
    ImportOptions[ImportOptions["ImportEquals"] = 2] = "ImportEquals";
    ImportOptions[ImportOptions["ExportFrom"] = 4] = "ExportFrom";
    ImportOptions[ImportOptions["DynamicImport"] = 8] = "DynamicImport";
    ImportOptions[ImportOptions["Require"] = 16] = "Require";
    ImportOptions[ImportOptions["All"] = 31] = "All";
    ImportOptions[ImportOptions["AllImports"] = 27] = "AllImports";
    ImportOptions[ImportOptions["AllStaticImports"] = 3] = "AllStaticImports";
    ImportOptions[ImportOptions["AllDynamic"] = 24] = "AllDynamic";
    ImportOptions[ImportOptions["AllRequireLike"] = 18] = "AllRequireLike";
})(ImportOptions = exports.ImportOptions || (exports.ImportOptions = {}));
function findImports(sourceFile, options) {
    return new ImportFinder(sourceFile, options).find();
}
exports.findImports = findImports;
var ImportFinder = (function () {
    function ImportFinder(_sourceFile, _options) {
        var _this = this;
        this._sourceFile = _sourceFile;
        this._options = _options;
        this._result = [];
        this._findNested = function (node) {
            if (node_1.isCallExpression(node)) {
                if (node.arguments.length === 1 &&
                    (node.expression.kind === ts.SyntaxKind.ImportKeyword && _this._options & 8 ||
                        _this._options & 16 && node.expression.kind === ts.SyntaxKind.Identifier &&
                            node.expression.text === 'require'))
                    _this._addImport(node.arguments[0]);
            }
            else if (node_1.isImportTypeNode(node) && node_1.isLiteralTypeNode(node.argument) && _this._options & 32) {
                _this._addImport(node.argument.literal);
            }
            ts.forEachChild(node, _this._findNested);
        };
    }
    ImportFinder.prototype.find = function () {
        if (this._sourceFile.isDeclarationFile)
            this._options &= ~24;
        this._findImports(this._sourceFile.statements);
        return this._result;
    };
    ImportFinder.prototype._findImports = function (statements) {
        for (var _i = 0, statements_1 = statements; _i < statements_1.length; _i++) {
            var statement = statements_1[_i];
            if (node_1.isImportDeclaration(statement)) {
                if (this._options & 1)
                    this._addImport(statement.moduleSpecifier);
            }
            else if (node_1.isImportEqualsDeclaration(statement)) {
                if (this._options & 2 &&
                    statement.moduleReference.kind === ts.SyntaxKind.ExternalModuleReference &&
                    statement.moduleReference.expression !== undefined)
                    this._addImport(statement.moduleReference.expression);
            }
            else if (node_1.isExportDeclaration(statement)) {
                if (statement.moduleSpecifier !== undefined && this._options & 4)
                    this._addImport(statement.moduleSpecifier);
            }
            else if (node_1.isModuleDeclaration(statement) &&
                this._options & (3 | 4) &&
                statement.body !== undefined && statement.name.kind === ts.SyntaxKind.StringLiteral &&
                ts.isExternalModule(this._sourceFile)) {
                this._findImports(statement.body.statements);
            }
            else if (this._options & 56) {
                ts.forEachChild(statement, this._findNested);
            }
        }
    };
    ImportFinder.prototype._addImport = function (expression) {
        if (node_1.isTextualLiteral(expression))
            this._result.push(expression);
    };
    return ImportFinder;
}());
function isStatementInAmbientContext(node) {
    while (node.flags & ts.NodeFlags.NestedNamespace)
        node = node.parent;
    return hasModifier(node.modifiers, ts.SyntaxKind.DeclareKeyword) || isAmbientModuleBlock(node.parent);
}
exports.isStatementInAmbientContext = isStatementInAmbientContext;
function isAmbientModuleBlock(node) {
    while (node.kind === ts.SyntaxKind.ModuleBlock) {
        do
            node = node.parent;
        while (node.flags & ts.NodeFlags.NestedNamespace);
        if (hasModifier(node.modifiers, ts.SyntaxKind.DeclareKeyword))
            return true;
        node = node.parent;
    }
    return false;
}
exports.isAmbientModuleBlock = isAmbientModuleBlock;
function getIIFE(func) {
    var node = func.parent;
    while (node.kind === ts.SyntaxKind.ParenthesizedExpression)
        node = node.parent;
    return node_1.isCallExpression(node) && func.end <= node.expression.end ? node : undefined;
}
exports.getIIFE = getIIFE;
function isStrictCompilerOptionEnabled(options, option) {
    return (options.strict ? options[option] !== false : options[option] === true) &&
        (option !== 'strictPropertyInitialization' || isStrictCompilerOptionEnabled(options, 'strictNullChecks'));
}
exports.isStrictCompilerOptionEnabled = isStrictCompilerOptionEnabled;
function isCompilerOptionEnabled(options, option) {
    switch (option) {
        case 'stripInternal':
            return options.stripInternal === true && isCompilerOptionEnabled(options, 'declaration');
        case 'declaration':
            return options.declaration || isCompilerOptionEnabled(options, 'composite');
        case 'noImplicitAny':
        case 'noImplicitThis':
        case 'strictNullChecks':
        case 'strictFunctionTypes':
        case 'strictPropertyInitialization':
        case 'alwaysStrict':
            return isStrictCompilerOptionEnabled(options, option);
    }
    return options[option] === true;
}
exports.isCompilerOptionEnabled = isCompilerOptionEnabled;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXRpbC5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbInV0aWwudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7O0FBQUEsK0JBQWlDO0FBRWpDLDBDQUcyQjtBQUczQix5REFBK0I7QUFFL0IsU0FBZ0IsY0FBYyxDQUEwQixJQUFhLEVBQUUsSUFBTyxFQUFFLFVBQTBCO0lBQ3RHLEtBQW9CLFVBQTRCLEVBQTVCLEtBQUEsSUFBSSxDQUFDLFdBQVcsQ0FBQyxVQUFVLENBQUMsRUFBNUIsY0FBNEIsRUFBNUIsSUFBNEI7UUFBM0MsSUFBTSxLQUFLLFNBQUE7UUFDWixJQUFJLEtBQUssQ0FBQyxJQUFJLEtBQUssSUFBSTtZQUNuQixPQUFvQixLQUFLLENBQUM7S0FBQTtBQUN0QyxDQUFDO0FBSkQsd0NBSUM7QUFFRCxTQUFnQixXQUFXLENBQUMsSUFBbUI7SUFDM0MsT0FBTyxJQUFJLElBQUksRUFBRSxDQUFDLFVBQVUsQ0FBQyxVQUFVLElBQUksSUFBSSxJQUFJLEVBQUUsQ0FBQyxVQUFVLENBQUMsU0FBUyxDQUFDO0FBQy9FLENBQUM7QUFGRCxrQ0FFQztBQUVELFNBQWdCLFVBQVUsQ0FBQyxJQUFtQjtJQUMxQyxPQUFPLElBQUksSUFBSSxFQUFFLENBQUMsVUFBVSxDQUFDLFNBQVMsQ0FBQztBQUMzQyxDQUFDO0FBRkQsZ0NBRUM7QUFFRCxTQUFnQixnQkFBZ0IsQ0FBQyxJQUFtQjtJQUNoRCxPQUFPLElBQUksSUFBSSxFQUFFLENBQUMsVUFBVSxDQUFDLGVBQWUsSUFBSSxJQUFJLElBQUksRUFBRSxDQUFDLFVBQVUsQ0FBQyxjQUFjLENBQUM7QUFDekYsQ0FBQztBQUZELDRDQUVDO0FBRUQsU0FBZ0IsY0FBYyxDQUFDLElBQW1CO0lBQzlDLE9BQU8sSUFBSSxJQUFJLEVBQUUsQ0FBQyxVQUFVLENBQUMsYUFBYSxJQUFJLElBQUksSUFBSSxFQUFFLENBQUMsVUFBVSxDQUFDLFlBQVksQ0FBQztBQUNyRixDQUFDO0FBRkQsd0NBRUM7QUFFRCxTQUFn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