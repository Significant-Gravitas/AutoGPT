"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var ts = require("typescript");
function isAccessorDeclaration(node) {
    return node.kind === ts.SyntaxKind.GetAccessor ||
        node.kind === ts.SyntaxKind.SetAccessor;
}
exports.isAccessorDeclaration = isAccessorDeclaration;
function isArrayBindingPattern(node) {
    return node.kind === ts.SyntaxKind.ArrayBindingPattern;
}
exports.isArrayBindingPattern = isArrayBindingPattern;
function isArrayLiteralExpression(node) {
    return node.kind === ts.SyntaxKind.ArrayLiteralExpression;
}
exports.isArrayLiteralExpression = isArrayLiteralExpression;
function isArrayTypeNode(node) {
    return node.kind === ts.SyntaxKind.ArrayType;
}
exports.isArrayTypeNode = isArrayTypeNode;
function isArrowFunction(node) {
    return node.kind === ts.SyntaxKind.ArrowFunction;
}
exports.isArrowFunction = isArrowFunction;
function isAsExpression(node) {
    return node.kind === ts.SyntaxKind.AsExpression;
}
exports.isAsExpression = isAsExpression;
function isAssertionExpression(node) {
    return node.kind === ts.SyntaxKind.AsExpression ||
        node.kind === ts.SyntaxKind.TypeAssertionExpression;
}
exports.isAssertionExpression = isAssertionExpression;
function isAwaitExpression(node) {
    return node.kind === ts.SyntaxKind.AwaitExpression;
}
exports.isAwaitExpression = isAwaitExpression;
function isBinaryExpression(node) {
    return node.kind === ts.SyntaxKind.BinaryExpression;
}
exports.isBinaryExpression = isBinaryExpression;
function isBindingElement(node) {
    return node.kind === ts.SyntaxKind.BindingElement;
}
exports.isBindingElement = isBindingElement;
function isBindingPattern(node) {
    return node.kind === ts.SyntaxKind.ArrayBindingPattern ||
        node.kind === ts.SyntaxKind.ObjectBindingPattern;
}
exports.isBindingPattern = isBindingPattern;
function isBlock(node) {
    return node.kind === ts.SyntaxKind.Block;
}
exports.isBlock = isBlock;
function isBlockLike(node) {
    return node.statements !== undefined;
}
exports.isBlockLike = isBlockLike;
function isBreakOrContinueStatement(node) {
    return node.kind === ts.SyntaxKind.BreakStatement ||
        node.kind === ts.SyntaxKind.ContinueStatement;
}
exports.isBreakOrContinueStatement = isBreakOrContinueStatement;
function isBreakStatement(node) {
    return node.kind === ts.SyntaxKind.BreakStatement;
}
exports.isBreakStatement = isBreakStatement;
function isCallExpression(node) {
    return node.kind === ts.SyntaxKind.CallExpression;
}
exports.isCallExpression = isCallExpression;
function isCallLikeExpression(node) {
    switch (node.kind) {
        case ts.SyntaxKind.CallExpression:
        case ts.SyntaxKind.Decorator:
        case ts.SyntaxKind.JsxOpeningElement:
        case ts.SyntaxKind.JsxSelfClosingElement:
        case ts.SyntaxKind.NewExpression:
        case ts.SyntaxKind.TaggedTemplateExpression:
            return true;
        default:
            return false;
    }
}
exports.isCallLikeExpression = isCallLikeExpression;
function isCallSignatureDeclaration(node) {
    return node.kind === ts.SyntaxKind.CallSignature;
}
exports.isCallSignatureDeclaration = isCallSignatureDeclaration;
function isCaseBlock(node) {
    return node.kind === ts.SyntaxKind.CaseBlock;
}
exports.isCaseBlock = isCaseBlock;
function isCaseClause(node) {
    return node.kind === ts.SyntaxKind.CaseClause;
}
exports.isCaseClause = isCaseClause;
function isCaseOrDefaultClause(node) {
    return node.kind === ts.SyntaxKind.CaseClause ||
        node.kind === ts.SyntaxKind.DefaultClause;
}
exports.isCaseOrDefaultClause = isCaseOrDefaultClause;
function isCatchClause(node) {
    return node.kind === ts.SyntaxKind.CatchClause;
}
exports.isCatchClause = isCatchClause;
function isClassDeclaration(node) {
    return node.kind === ts.SyntaxKind.ClassDeclaration;
}
exports.isClassDeclaration = isClassDeclaration;
function isClassExpression(node) {
    return node.kind === ts.SyntaxKind.ClassExpression;
}
exports.isClassExpression = isClassExpression;
function isClassLikeDeclaration(node) {
    return node.kind === ts.SyntaxKind.ClassDeclaration ||
        node.kind === ts.SyntaxKind.ClassExpression;
}
exports.isClassLikeDeclaration = isClassLikeDeclaration;
function isCommaListExpression(node) {
    return node.kind === ts.SyntaxKind.CommaListExpression;
}
exports.isCommaListExpression = isCommaListExpression;
function isConditionalExpression(node) {
    return node.kind === ts.SyntaxKind.ConditionalExpression;
}
exports.isConditionalExpression = isConditionalExpression;
function isConditionalTypeNode(node) {
    return node.kind === ts.SyntaxKind.ConditionalType;
}
exports.isConditionalTypeNode = isConditionalTypeNode;
function isConstructorDeclaration(node) {
    return node.kind === ts.SyntaxKind.Constructor;
}
exports.isConstructorDeclaration = isConstructorDeclaration;
function isConstructorTypeNode(node) {
    return node.kind === ts.SyntaxKind.ConstructorType;
}
exports.isConstructorTypeNode = isConstructorTypeNode;
function isConstructSignatureDeclaration(node) {
    return node.kind === ts.SyntaxKind.ConstructSignature;
}
exports.isConstructSignatureDeclaration = isConstructSignatureDeclaration;
function isContinueStatement(node) {
    return node.kind === ts.SyntaxKind.ContinueStatement;
}
exports.isContinueStatement = isContinueStatement;
function isComputedPropertyName(node) {
    return node.kind === ts.SyntaxKind.ComputedPropertyName;
}
exports.isComputedPropertyName = isComputedPropertyName;
function isDebuggerStatement(node) {
    return node.kind === ts.SyntaxKind.DebuggerStatement;
}
exports.isDebuggerStatement = isDebuggerStatement;
function isDecorator(node) {
    return node.kind === ts.SyntaxKind.Decorator;
}
exports.isDecorator = isDecorator;
function isDefaultClause(node) {
    return node.kind === ts.SyntaxKind.DefaultClause;
}
exports.isDefaultClause = isDefaultClause;
function isDeleteExpression(node) {
    return node.kind === ts.SyntaxKind.DeleteExpression;
}
exports.isDeleteExpression = isDeleteExpression;
function isDoStatement(node) {
    return node.kind === ts.SyntaxKind.DoStatement;
}
exports.isDoStatement = isDoStatement;
function isElementAccessExpression(node) {
    return node.kind === ts.SyntaxKind.ElementAccessExpression;
}
exports.isElementAccessExpression = isElementAccessExpression;
function isEmptyStatement(node) {
    return node.kind === ts.SyntaxKind.EmptyStatement;
}
exports.isEmptyStatement = isEmptyStatement;
function isEntityName(node) {
    return node.kind === ts.SyntaxKind.Identifier || isQualifiedName(node);
}
exports.isEntityName = isEntityName;
function isEntityNameExpression(node) {
    return node.kind === ts.SyntaxKind.Identifier ||
        isPropertyAccessExpression(node) && isEntityNameExpression(node.expression);
}
exports.isEntityNameExpression = isEntityNameExpression;
function isEnumDeclaration(node) {
    return node.kind === ts.SyntaxKind.EnumDeclaration;
}
exports.isEnumDeclaration = isEnumDeclaration;
function isEnumMember(node) {
    return node.kind === ts.SyntaxKind.EnumMember;
}
exports.isEnumMember = isEnumMember;
function isExportAssignment(node) {
    return node.kind === ts.SyntaxKind.ExportAssignment;
}
exports.isExportAssignment = isExportAssignment;
function isExportDeclaration(node) {
    return node.kind === ts.SyntaxKind.ExportDeclaration;
}
exports.isExportDeclaration = isExportDeclaration;
function isExportSpecifier(node) {
    return node.kind === ts.SyntaxKind.ExportSpecifier;
}
exports.isExportSpecifier = isExportSpecifier;
function isExpression(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ArrayLiteralExpression:
        case ts.SyntaxKind.ArrowFunction:
        case ts.SyntaxKind.AsExpression:
        case ts.SyntaxKind.AwaitExpression:
        case ts.SyntaxKind.BinaryExpression:
        case ts.SyntaxKind.CallExpression:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.CommaListExpression:
        case ts.SyntaxKind.ConditionalExpression:
        case ts.SyntaxKind.DeleteExpression:
        case ts.SyntaxKind.ElementAccessExpression:
        case ts.SyntaxKind.FalseKeyword:
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.Identifier:
        case ts.SyntaxKind.JsxElement:
        case ts.SyntaxKind.JsxFragment:
        case ts.SyntaxKind.JsxExpression:
        case ts.SyntaxKind.JsxOpeningElement:
        case ts.SyntaxKind.JsxOpeningFragment:
        case ts.SyntaxKind.JsxSelfClosingElement:
        case ts.SyntaxKind.MetaProperty:
        case ts.SyntaxKind.NewExpression:
        case ts.SyntaxKind.NonNullExpression:
        case ts.SyntaxKind.NoSubstitutionTemplateLiteral:
        case ts.SyntaxKind.NullKeyword:
        case ts.SyntaxKind.NumericLiteral:
        case ts.SyntaxKind.ObjectLiteralExpression:
        case ts.SyntaxKind.OmittedExpression:
        case ts.SyntaxKind.ParenthesizedExpression:
        case ts.SyntaxKind.PostfixUnaryExpression:
        case ts.SyntaxKind.PrefixUnaryExpression:
        case ts.SyntaxKind.PropertyAccessExpression:
        case ts.SyntaxKind.RegularExpressionLiteral:
        case ts.SyntaxKind.SpreadElement:
        case ts.SyntaxKind.StringLiteral:
        case ts.SyntaxKind.SuperKeyword:
        case ts.SyntaxKind.TaggedTemplateExpression:
        case ts.SyntaxKind.TemplateExpression:
        case ts.SyntaxKind.ThisKeyword:
        case ts.SyntaxKind.TrueKeyword:
        case ts.SyntaxKind.TypeAssertionExpression:
        case ts.SyntaxKind.TypeOfExpression:
        case ts.SyntaxKind.VoidExpression:
        case ts.SyntaxKind.YieldExpression:
            return true;
        default:
            return false;
    }
}
exports.isExpression = isExpression;
function isExpressionStatement(node) {
    return node.kind === ts.SyntaxKind.ExpressionStatement;
}
exports.isExpressionStatement = isExpressionStatement;
function isExpressionWithTypeArguments(node) {
    return node.kind === ts.SyntaxKind.ExpressionWithTypeArguments;
}
exports.isExpressionWithTypeArguments = isExpressionWithTypeArguments;
function isExternalModuleReference(node) {
    return node.kind === ts.SyntaxKind.ExternalModuleReference;
}
exports.isExternalModuleReference = isExternalModuleReference;
function isForInStatement(node) {
    return node.kind === ts.SyntaxKind.ForInStatement;
}
exports.isForInStatement = isForInStatement;
function isForInOrOfStatement(node) {
    return node.kind === ts.SyntaxKind.ForOfStatement || node.kind === ts.SyntaxKind.ForInStatement;
}
exports.isForInOrOfStatement = isForInOrOfStatement;
function isForOfStatement(node) {
    return node.kind === ts.SyntaxKind.ForOfStatement;
}
exports.isForOfStatement = isForOfStatement;
function isForStatement(node) {
    return node.kind === ts.SyntaxKind.ForStatement;
}
exports.isForStatement = isForStatement;
function isFunctionDeclaration(node) {
    return node.kind === ts.SyntaxKind.FunctionDeclaration;
}
exports.isFunctionDeclaration = isFunctionDeclaration;
function isFunctionExpression(node) {
    return node.kind === ts.SyntaxKind.FunctionExpression;
}
exports.isFunctionExpression = isFunctionExpression;
function isFunctionTypeNode(node) {
    return node.kind === ts.SyntaxKind.FunctionType;
}
exports.isFunctionTypeNode = isFunctionTypeNode;
function isGetAccessorDeclaration(node) {
    return node.kind === ts.SyntaxKind.GetAccessor;
}
exports.isGetAccessorDeclaration = isGetAccessorDeclaration;
function isIdentifier(node) {
    return node.kind === ts.SyntaxKind.Identifier;
}
exports.isIdentifier = isIdentifier;
function isIfStatement(node) {
    return node.kind === ts.SyntaxKind.IfStatement;
}
exports.isIfStatement = isIfStatement;
function isImportClause(node) {
    return node.kind === ts.SyntaxKind.ImportClause;
}
exports.isImportClause = isImportClause;
function isImportDeclaration(node) {
    return node.kind === ts.SyntaxKind.ImportDeclaration;
}
exports.isImportDeclaration = isImportDeclaration;
function isImportEqualsDeclaration(node) {
    return node.kind === ts.SyntaxKind.ImportEqualsDeclaration;
}
exports.isImportEqualsDeclaration = isImportEqualsDeclaration;
function isImportSpecifier(node) {
    return node.kind === ts.SyntaxKind.ImportSpecifier;
}
exports.isImportSpecifier = isImportSpecifier;
function isIndexedAccessTypeNode(node) {
    return node.kind === ts.SyntaxKind.IndexedAccessType;
}
exports.isIndexedAccessTypeNode = isIndexedAccessTypeNode;
function isIndexSignatureDeclaration(node) {
    return node.kind === ts.SyntaxKind.IndexSignature;
}
exports.isIndexSignatureDeclaration = isIndexSignatureDeclaration;
function isInferTypeNode(node) {
    return node.kind === ts.SyntaxKind.InferType;
}
exports.isInferTypeNode = isInferTypeNode;
function isInterfaceDeclaration(node) {
    return node.kind === ts.SyntaxKind.InterfaceDeclaration;
}
exports.isInterfaceDeclaration = isInterfaceDeclaration;
function isIntersectionTypeNode(node) {
    return node.kind === ts.SyntaxKind.IntersectionType;
}
exports.isIntersectionTypeNode = isIntersectionTypeNode;
function isIterationStatement(node) {
    switch (node.kind) {
        case ts.SyntaxKind.ForStatement:
        case ts.SyntaxKind.ForOfStatement:
        case ts.SyntaxKind.ForInStatement:
        case ts.SyntaxKind.WhileStatement:
        case ts.SyntaxKind.DoStatement:
            return true;
        default:
            return false;
    }
}
exports.isIterationStatement = isIterationStatement;
function isJsDoc(node) {
    return node.kind === ts.SyntaxKind.JSDocComment;
}
exports.isJsDoc = isJsDoc;
function isJsxAttribute(node) {
    return node.kind === ts.SyntaxKind.JsxAttribute;
}
exports.isJsxAttribute = isJsxAttribute;
function isJsxAttributeLike(node) {
    return node.kind === ts.SyntaxKind.JsxAttribute ||
        node.kind === ts.SyntaxKind.JsxSpreadAttribute;
}
exports.isJsxAttributeLike = isJsxAttributeLike;
function isJsxAttributes(node) {
    return node.kind === ts.SyntaxKind.JsxAttributes;
}
exports.isJsxAttributes = isJsxAttributes;
function isJsxClosingElement(node) {
    return node.kind === ts.SyntaxKind.JsxClosingElement;
}
exports.isJsxClosingElement = isJsxClosingElement;
function isJsxClosingFragment(node) {
    return node.kind === ts.SyntaxKind.JsxClosingFragment;
}
exports.isJsxClosingFragment = isJsxClosingFragment;
function isJsxElement(node) {
    return node.kind === ts.SyntaxKind.JsxElement;
}
exports.isJsxElement = isJsxElement;
function isJsxExpression(node) {
    return node.kind === ts.SyntaxKind.JsxExpression;
}
exports.isJsxExpression = isJsxExpression;
function isJsxFramgment(node) {
    return isJsxFragment(node);
}
exports.isJsxFramgment = isJsxFramgment;
function isJsxFragment(node) {
    return node.kind === ts.SyntaxKind.JsxFragment;
}
exports.isJsxFragment = isJsxFragment;
function isJsxOpeningElement(node) {
    return node.kind === ts.SyntaxKind.JsxOpeningElement;
}
exports.isJsxOpeningElement = isJsxOpeningElement;
function isJsxOpeningFragment(node) {
    return node.kind === ts.SyntaxKind.JsxOpeningFragment;
}
exports.isJsxOpeningFragment = isJsxOpeningFragment;
function isJsxOpeningLikeElement(node) {
    return node.kind === ts.SyntaxKind.JsxOpeningElement ||
        node.kind === ts.SyntaxKind.JsxSelfClosingElement;
}
exports.isJsxOpeningLikeElement = isJsxOpeningLikeElement;
function isJsxSelfClosingElement(node) {
    return node.kind === ts.SyntaxKind.JsxSelfClosingElement;
}
exports.isJsxSelfClosingElement = isJsxSelfClosingElement;
function isJsxSpreadAttribute(node) {
    return node.kind === ts.SyntaxKind.JsxSpreadAttribute;
}
exports.isJsxSpreadAttribute = isJsxSpreadAttribute;
function isJsxText(node) {
    return node.kind === ts.SyntaxKind.JsxText;
}
exports.isJsxText = isJsxText;
function isLabeledStatement(node) {
    return node.kind === ts.SyntaxKind.LabeledStatement;
}
exports.isLabeledStatement = isLabeledStatement;
function isLiteralExpression(node) {
    return node.kind >= ts.SyntaxKind.FirstLiteralToken &&
        node.kind <= ts.SyntaxKind.LastLiteralToken;
}
exports.isLiteralExpression = isLiteralExpression;
function isLiteralTypeNode(node) {
    return node.kind === ts.SyntaxKind.LiteralType;
}
exports.isLiteralTypeNode = isLiteralTypeNode;
function isMappedTypeNode(node) {
    return node.kind === ts.SyntaxKind.MappedType;
}
exports.isMappedTypeNode = isMappedTypeNode;
function isMetaProperty(node) {
    return node.kind === ts.SyntaxKind.MetaProperty;
}
exports.isMetaProperty = isMetaProperty;
function isMethodDeclaration(node) {
    return node.kind === ts.SyntaxKind.MethodDeclaration;
}
exports.isMethodDeclaration = isMethodDeclaration;
function isMethodSignature(node) {
    return node.kind === ts.SyntaxKind.MethodSignature;
}
exports.isMethodSignature = isMethodSignature;
function isModuleBlock(node) {
    return node.kind === ts.SyntaxKind.ModuleBlock;
}
exports.isModuleBlock = isModuleBlock;
function isModuleDeclaration(node) {
    return node.kind === ts.SyntaxKind.ModuleDeclaration;
}
exports.isModuleDeclaration = isModuleDeclaration;
function isNamedExports(node) {
    return node.kind === ts.SyntaxKind.NamedExports;
}
exports.isNamedExports = isNamedExports;
function isNamedImports(node) {
    return node.kind === ts.SyntaxKind.NamedImports;
}
exports.isNamedImports = isNamedImports;
function isNamespaceDeclaration(node) {
    return isModuleDeclaration(node) &&
        node.name.kind === ts.SyntaxKind.Identifier &&
        node.body !== undefined &&
        (node.body.kind === ts.SyntaxKind.ModuleBlock ||
            isNamespaceDeclaration(node.body));
}
exports.isNamespaceDeclaration = isNamespaceDeclaration;
function isNamespaceImport(node) {
    return node.kind === ts.SyntaxKind.NamespaceImport;
}
exports.isNamespaceImport = isNamespaceImport;
function isNamespaceExportDeclaration(node) {
    return node.kind === ts.SyntaxKind.NamespaceExportDeclaration;
}
exports.isNamespaceExportDeclaration = isNamespaceExportDeclaration;
function isNewExpression(node) {
    return node.kind === ts.SyntaxKind.NewExpression;
}
exports.isNewExpression = isNewExpression;
function isNonNullExpression(node) {
    return node.kind === ts.SyntaxKind.NonNullExpression;
}
exports.isNonNullExpression = isNonNullExpression;
function isNoSubstitutionTemplateLiteral(node) {
    return node.kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral;
}
exports.isNoSubstitutionTemplateLiteral = isNoSubstitutionTemplateLiteral;
function isNumericLiteral(node) {
    return node.kind === ts.SyntaxKind.NumericLiteral;
}
exports.isNumericLiteral = isNumericLiteral;
function isObjectBindingPattern(node) {
    return node.kind === ts.SyntaxKind.ObjectBindingPattern;
}
exports.isObjectBindingPattern = isObjectBindingPattern;
function isObjectLiteralExpression(node) {
    return node.kind === ts.SyntaxKind.ObjectLiteralExpression;
}
exports.isObjectLiteralExpression = isObjectLiteralExpression;
function isOmittedExpression(node) {
    return node.kind === ts.SyntaxKind.OmittedExpression;
}
exports.isOmittedExpression = isOmittedExpression;
function isParameterDeclaration(node) {
    return node.kind === ts.SyntaxKind.Parameter;
}
exports.isParameterDeclaration = isParameterDeclaration;
function isParenthesizedExpression(node) {
    return node.kind === ts.SyntaxKind.ParenthesizedExpression;
}
exports.isParenthesizedExpression = isParenthesizedExpression;
function isParenthesizedTypeNode(node) {
    return node.kind === ts.SyntaxKind.ParenthesizedType;
}
exports.isParenthesizedTypeNode = isParenthesizedTypeNode;
function isPostfixUnaryExpression(node) {
    return node.kind === ts.SyntaxKind.PostfixUnaryExpression;
}
exports.isPostfixUnaryExpression = isPostfixUnaryExpression;
function isPrefixUnaryExpression(node) {
    return node.kind === ts.SyntaxKind.PrefixUnaryExpression;
}
exports.isPrefixUnaryExpression = isPrefixUnaryExpression;
function isPropertyAccessExpression(node) {
    return node.kind === ts.SyntaxKind.PropertyAccessExpression;
}
exports.isPropertyAccessExpression = isPropertyAccessExpression;
function isPropertyAssignment(node) {
    return node.kind === ts.SyntaxKind.PropertyAssignment;
}
exports.isPropertyAssignment = isPropertyAssignment;
function isPropertyDeclaration(node) {
    return node.kind === ts.SyntaxKind.PropertyDeclaration;
}
exports.isPropertyDeclaration = isPropertyDeclaration;
function isPropertySignature(node) {
    return node.kind === ts.SyntaxKind.PropertySignature;
}
exports.isPropertySignature = isPropertySignature;
function isQualifiedName(node) {
    return node.kind === ts.SyntaxKind.QualifiedName;
}
exports.isQualifiedName = isQualifiedName;
function isRegularExpressionLiteral(node) {
    return node.kind === ts.SyntaxKind.RegularExpressionLiteral;
}
exports.isRegularExpressionLiteral = isRegularExpressionLiteral;
function isReturnStatement(node) {
    return node.kind === ts.SyntaxKind.ReturnStatement;
}
exports.isReturnStatement = isReturnStatement;
function isSetAccessorDeclaration(node) {
    return node.kind === ts.SyntaxKind.SetAccessor;
}
exports.isSetAccessorDeclaration = isSetAccessorDeclaration;
function isShorthandPropertyAssignment(node) {
    return node.kind === ts.SyntaxKind.ShorthandPropertyAssignment;
}
exports.isShorthandPropertyAssignment = isShorthandPropertyAssignment;
function isSignatureDeclaration(node) {
    return node.parameters !== undefined;
}
exports.isSignatureDeclaration = isSignatureDeclaration;
function isSourceFile(node) {
    return node.kind === ts.SyntaxKind.SourceFile;
}
exports.isSourceFile = isSourceFile;
function isSpreadAssignment(node) {
    return node.kind === ts.SyntaxKind.SpreadAssignment;
}
exports.isSpreadAssignment = isSpreadAssignment;
function isSpreadElement(node) {
    return node.kind === ts.SyntaxKind.SpreadElement;
}
exports.isSpreadElement = isSpreadElement;
function isStringLiteral(node) {
    return node.kind === ts.SyntaxKind.StringLiteral;
}
exports.isStringLiteral = isStringLiteral;
function isSwitchStatement(node) {
    return node.kind === ts.SyntaxKind.SwitchStatement;
}
exports.isSwitchStatement = isSwitchStatement;
function isSyntaxList(node) {
    return node.kind === ts.SyntaxKind.SyntaxList;
}
exports.isSyntaxList = isSyntaxList;
function isTaggedTemplateExpression(node) {
    return node.kind === ts.SyntaxKind.TaggedTemplateExpression;
}
exports.isTaggedTemplateExpression = isTaggedTemplateExpression;
function isTemplateExpression(node) {
    return node.kind === ts.SyntaxKind.TemplateExpression;
}
exports.isTemplateExpression = isTemplateExpression;
function isTemplateLiteral(node) {
    return node.kind === ts.SyntaxKind.TemplateExpression ||
        node.kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral;
}
exports.isTemplateLiteral = isTemplateLiteral;
function isTextualLiteral(node) {
    return node.kind === ts.SyntaxKind.StringLiteral ||
        node.kind === ts.SyntaxKind.NoSubstitutionTemplateLiteral;
}
exports.isTextualLiteral = isTextualLiteral;
function isThrowStatement(node) {
    return node.kind === ts.SyntaxKind.ThrowStatement;
}
exports.isThrowStatement = isThrowStatement;
function isTryStatement(node) {
    return node.kind === ts.SyntaxKind.TryStatement;
}
exports.isTryStatement = isTryStatement;
function isTupleTypeNode(node) {
    return node.kind === ts.SyntaxKind.TupleType;
}
exports.isTupleTypeNode = isTupleTypeNode;
function isTypeAliasDeclaration(node) {
    return node.kind === ts.SyntaxKind.TypeAliasDeclaration;
}
exports.isTypeAliasDeclaration = isTypeAliasDeclaration;
function isTypeAssertion(node) {
    return node.kind === ts.SyntaxKind.TypeAssertionExpression;
}
exports.isTypeAssertion = isTypeAssertion;
function isTypeLiteralNode(node) {
    return node.kind === ts.SyntaxKind.TypeLiteral;
}
exports.isTypeLiteralNode = isTypeLiteralNode;
function isTypeOfExpression(node) {
    return node.kind === ts.SyntaxKind.TypeOfExpression;
}
exports.isTypeOfExpression = isTypeOfExpression;
function isTypeOperatorNode(node) {
    return node.kind === ts.SyntaxKind.TypeOperator;
}
exports.isTypeOperatorNode = isTypeOperatorNode;
function isTypeParameterDeclaration(node) {
    return node.kind === ts.SyntaxKind.TypeParameter;
}
exports.isTypeParameterDeclaration = isTypeParameterDeclaration;
function isTypePredicateNode(node) {
    return node.kind === ts.SyntaxKind.TypePredicate;
}
exports.isTypePredicateNode = isTypePredicateNode;
function isTypeReferenceNode(node) {
    return node.kind === ts.SyntaxKind.TypeReference;
}
exports.isTypeReferenceNode = isTypeReferenceNode;
function isTypeQueryNode(node) {
    return node.kind === ts.SyntaxKind.TypeQuery;
}
exports.isTypeQueryNode = isTypeQueryNode;
function isUnionTypeNode(node) {
    return node.kind === ts.SyntaxKind.UnionType;
}
exports.isUnionTypeNode = isUnionTypeNode;
function isVariableDeclaration(node) {
    return node.kind === ts.SyntaxKind.VariableDeclaration;
}
exports.isVariableDeclaration = isVariableDeclaration;
function isVariableStatement(node) {
    return node.kind === ts.SyntaxKind.VariableStatement;
}
exports.isVariableStatement = isVariableStatement;
function isVariableDeclarationList(node) {
    return node.kind === ts.SyntaxKind.VariableDeclarationList;
}
exports.isVariableDeclarationList = isVariableDeclarationList;
function isVoidExpression(node) {
    return node.kind === ts.SyntaxKind.VoidExpression;
}
exports.isVoidExpression = isVoidExpression;
function isWhileStatement(node) {
    return node.kind === ts.SyntaxKind.WhileStatement;
}
exports.isWhileStatement = isWhileStatement;
function isWithStatement(node) {
    return node.kind === ts.SyntaxKind.WithStatement;
}
exports.isWithStatement = isWithStatement;
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoibm9kZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIm5vZGUudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6Ijs7QUFBQSwrQkFBaUM7QUFFakMsU0FBZ0IscUJBQXFCLENBQUMsSUFBYTtJQUMvQyxPQUFPLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxXQUFXO1FBQzFDLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxXQUFXLENBQUM7QUFDaEQsQ0FBQztBQUhELHNEQUdDO0FBRUQsU0FBZ0IscUJBQXFCLENBQUMsSUFBYTtJQUMvQyxPQUFPLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxtQkFBbUIsQ0FBQztBQUMzRCxDQUFDO0FBRkQsc0RBRUM7QUFFRCxTQUFnQix3QkFBd0IsQ0FBQyxJQUFhO0lBQ2xELE9BQU8sSUFBSSxDQUFDLElBQUksS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLHNCQUFzQixDQUFDO0FBQzlELENBQUM7QUFGRCw0REFFQztBQUVELFNBQWdCLGVBQWUsQ0FBQyxJQUFhO0lBQ3pDLE9BQU8sSUFBSSxDQUFDLElBQUksS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLFNBQVMsQ0FBQztBQUNqRCxDQUFDO0FBRkQsMENBRUM7QUFFRCxTQUFnQixlQUFlLENBQUMsSUFBYTtJQUN6QyxPQUFPLElBQUksQ0FBQyxJQUFJLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxhQUFhLENBQUM7QUFDckQsQ0FBQztBQUZELDBDQUVDO0FBRUQsU0FBZ0IsY0FBYyxDQUFDLElBQWE7SUFDeEMsT0FBTyxJQUFJLENBQUMsSUFBSSxLQUFLLEVBQUUsQ0FBQyxVQUFVLENBQUMsWUFBWSxDQUFDO0FBQ3BELENBQUM7QUFGRCx3Q0FFQztBQUVELFNBQWdCLHFCQUFxQixDQUFDLElBQWE7SUFDL0MsT0FBTyxJQUFJLENBQUMsSUFBSSxLQUFLLEVBQUUsQ0FBQyxVQUFVLENBQUMs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