"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tslib_1 = require("tslib");
var util_1 = require("./util");
var ts = require("typescript");
var DeclarationDomain;
(function (DeclarationDomain) {
    DeclarationDomain[DeclarationDomain["Namespace"] = 1] = "Namespace";
    DeclarationDomain[DeclarationDomain["Type"] = 2] = "Type";
    DeclarationDomain[DeclarationDomain["Value"] = 4] = "Value";
    DeclarationDomain[DeclarationDomain["Import"] = 8] = "Import";
    DeclarationDomain[DeclarationDomain["Any"] = 7] = "Any";
})(DeclarationDomain = exports.DeclarationDomain || (exports.DeclarationDomain = {}));
var UsageDomain;
(function (UsageDomain) {
    UsageDomain[UsageDomain["Namespace"] = 1] = "Namespace";
    UsageDomain[UsageDomain["Type"] = 2] = "Type";
    UsageDomain[UsageDomain["Value"] = 4] = "Value";
    UsageDomain[UsageDomain["ValueOrNamespace"] = 5] = "ValueOrNamespace";
    UsageDomain[UsageDomain["Any"] = 7] = "Any";
    UsageDomain[UsageDomain["TypeQuery"] = 8] = "TypeQuery";
})(UsageDomain = exports.UsageDomain || (exports.UsageDomain = {}));
function getUsageDomain(node) {
    var parent = node.parent;
    switch (parent.kind) {
        case ts.SyntaxKind.TypeReference:
            return 2;
        case ts.SyntaxKind.ExpressionWithTypeArguments:
            return parent.parent.token === ts.SyntaxKind.ImplementsKeyword ||
                parent.parent.parent.kind === ts.SyntaxKind.InterfaceDeclaration
                ? 2
                : 4;
        case ts.SyntaxKind.TypeQuery:
            return 5 | 8;
        case ts.SyntaxKind.QualifiedName:
            if (parent.left === node) {
                if (getEntityNameParent(parent).kind === ts.SyntaxKind.TypeQuery)
                    return 1 | 8;
                return 1;
            }
            break;
        case ts.SyntaxKind.ExportSpecifier:
            if (parent.propertyName === undefined ||
                parent.propertyName === node)
                return 7;
            break;
        case ts.SyntaxKind.ExportAssignment:
            return 7;
        case ts.SyntaxKind.BindingElement:
            if (parent.initializer === node)
                return 5;
            break;
        case ts.SyntaxKind.Parameter:
        case ts.SyntaxKind.EnumMember:
        case ts.SyntaxKind.PropertyDeclaration:
        case ts.SyntaxKind.VariableDeclaration:
        case ts.SyntaxKind.PropertyAssignment:
        case ts.SyntaxKind.PropertyAccessExpression:
        case ts.SyntaxKind.ImportEqualsDeclaration:
            if (parent.name !== node)
                return 5;
            break;
        case ts.SyntaxKind.JsxAttribute:
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.FunctionExpression:
        case ts.SyntaxKind.NamespaceImport:
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.ClassExpression:
        case ts.SyntaxKind.ModuleDeclaration:
        case ts.SyntaxKind.MethodDeclaration:
        case ts.SyntaxKind.EnumDeclaration:
        case ts.SyntaxKind.GetAccessor:
        case ts.SyntaxKind.SetAccessor:
        case ts.SyntaxKind.LabeledStatement:
        case ts.SyntaxKind.BreakStatement:
        case ts.SyntaxKind.ContinueStatement:
        case ts.SyntaxKind.ImportClause:
        case ts.SyntaxKind.ImportSpecifier:
        case ts.SyntaxKind.TypePredicate:
        case ts.SyntaxKind.MethodSignature:
        case ts.SyntaxKind.PropertySignature:
        case ts.SyntaxKind.NamespaceExportDeclaration:
        case ts.SyntaxKind.InterfaceDeclaration:
        case ts.SyntaxKind.TypeAliasDeclaration:
        case ts.SyntaxKind.TypeParameter:
            break;
        default:
            return 5;
    }
}
exports.getUsageDomain = getUsageDomain;
function getDeclarationDomain(node) {
    switch (node.parent.kind) {
        case ts.SyntaxKind.TypeParameter:
        case ts.SyntaxKind.InterfaceDeclaration:
        case ts.SyntaxKind.TypeAliasDeclaration:
            return 2;
        case ts.SyntaxKind.ClassDeclaration:
        case ts.SyntaxKind.ClassExpression:
            return 2 | 4;
        case ts.SyntaxKind.EnumDeclaration:
            return 7;
        case ts.SyntaxKind.NamespaceImport:
        case ts.SyntaxKind.ImportClause:
            return 7 | 8;
        case ts.SyntaxKind.ImportEqualsDeclaration:
        case ts.SyntaxKind.ImportSpecifier:
            return node.parent.name === node
                ? 7 | 8
                : undefined;
        case ts.SyntaxKind.ModuleDeclaration:
            return 1;
        case ts.SyntaxKind.Parameter:
            if (node.parent.parent.kind === ts.SyntaxKind.IndexSignature)
                return;
        case ts.SyntaxKind.BindingElement:
        case ts.SyntaxKind.VariableDeclaration:
            return node.parent.name === node ? 4 : undefined;
        case ts.SyntaxKind.FunctionDeclaration:
        case ts.SyntaxKind.FunctionExpression:
            return 4;
    }
}
exports.getDeclarationDomain = getDeclarationDomain;
function collectVariableUsage(sourceFile) {
    return new UsageWalker().getUsage(sourceFile);
}
exports.collectVariableUsage = collectVariableUsage;
var AbstractScope = (function () {
    function AbstractScope(_global) {
        this._global = _global;
        this._variables = new Map();
        this._uses = [];
        this._namespaceScopes = undefined;
        this._enumScopes = undefined;
    }
    AbstractScope.prototype.addVariable = function (identifier, name, blockScoped, exported, domain) {
        var variables = this._getDestinationScope(blockScoped).getVariables();
        var declaration = {
            domain: domain,
            exported: exported,
            declaration: name,
        };
        var variable = variables.get(identifier);
        if (variable === undefined) {
            variables.set(identifier, {
                domain: domain,
                declarations: [declaration],
                uses: [],
            });
        }
        else {
            variable.domain |= domain;
            variable.declarations.push(declaration);
        }
    };
    AbstractScope.prototype.addUse = function (use) {
        this._uses.push(use);
    };
    AbstractScope.prototype.getVariables = function () {
        return this._variables;
    };
    AbstractScope.prototype.getFunctionScope = function () {
        return this;
    };
    AbstractScope.prototype.end = function (cb) {
        var _this = this;
        if (this._namespaceScopes !== undefined)
            this._namespaceScopes.forEach(function (value) { return value.finish(cb); });
        this._namespaceScopes = this._enumScopes = undefined;
        this._applyUses();
        this._variables.forEach(function (variable) {
            for (var _i = 0, _a = variable.declarations; _i < _a.length; _i++) {
                var declaration = _a[_i];
                var result = {
                    declarations: [],
                    domain: declaration.domain,
                    exported: declaration.exported,
                    inGlobalScope: _this._global,
                    uses: [],
                };
                for (var _b = 0, _c = variable.declarations; _b < _c.length; _b++) {
                    var other = _c[_b];
                    if (other.domain & declaration.domain)
                        result.declarations.push(other.declaration);
                }
                for (var _d = 0, _e = variable.uses; _d < _e.length; _d++) {
                    var use = _e[_d];
                    if (use.domain & declaration.domain)
                        result.uses.push(use);
                }
                cb(result, declaration.declaration, _this);
            }
        });
    };
    AbstractScope.prototype.markExported = function (_name) { };
    AbstractScope.prototype.createOrReuseNamespaceScope = function (name, _exported, ambient, hasExportStatement) {
        var scope;
        if (this._namespaceScopes === undefined) {
            this._namespaceScopes = new Map();
        }
        else {
            scope = this._namespaceScopes.get(name);
        }
        if (scope === undefined) {
            scope = new NamespaceScope(ambient, hasExportStatement, this);
            this._namespaceScopes.set(name, scope);
        }
        else {
            scope.refresh(ambient, hasExportStatement);
        }
        return scope;
    };
    AbstractScope.prototype.createOrReuseEnumScope = function (name, _exported) {
        var scope;
        if (this._enumScopes === undefined) {
            this._enumScopes = new Map();
        }
        else {
            scope = this._enumScopes.get(name);
        }
        if (scope === undefined) {
            scope = new EnumScope(this);
            this._enumScopes.set(name, scope);
        }
        return scope;
    };
    AbstractScope.prototype._applyUses = function () {
        for (var _i = 0, _a = this._uses; _i < _a.length; _i++) {
            var use = _a[_i];
            if (!this._applyUse(use))
                this._addUseToParent(use);
        }
        this._uses = [];
    };
    AbstractScope.prototype._applyUse = function (use, variables) {
        if (variables === void 0) { variables = this._variables; }
        var variable = variables.get(util_1.getIdentifierText(use.location));
        if (variable === undefined || (variable.domain & use.domain) === 0)
            return false;
        variable.uses.push(use);
        return true;
    };
    AbstractScope.prototype._getDestinationScope = function (_blockScoped) {
        return this;
    };
    AbstractScope.prototype._addUseToParent = function (_use) { };
    return AbstractScope;
}());
var RootScope = (function (_super) {
    tslib_1.__extends(RootScope, _super);
    function RootScope(_exportAll, global) {
        var _this = _super.call(this, global) || this;
        _this._exportAll = _exportAll;
        _this._exports = undefined;
        _this._innerScope = new NonRootScope(_this);
        return _this;
    }
    RootScope.prototype.addVariable = function (identifier, name, blockScoped, exported, domain) {
        if (domain & 8)
            return _super.prototype.addVariable.call(this, identifier, name, blockScoped, exported, domain);
        return this._innerScope.addVariable(identifier, name, blockScoped, exported, domain);
    };
    RootScope.prototype.addUse = function (use, origin) {
        if (origin === this._innerScope)
            return _super.prototype.addUse.call(this, use);
        return this._innerScope.addUse(use);
    };
    RootScope.prototype.markExported = function (id) {
        var text = util_1.getIdentifierText(id);
        if (this._exports === undefined) {
            this._exports = [text];
        }
        else {
            this._exports.push(text);
        }
    };
    RootScope.prototype.end = function (cb) {
        var _this = this;
        this._innerScope.end(function (value, key) {
            value.exported = value.exported || _this._exportAll
                || _this._exports !== undefined && _this._exports.indexOf(util_1.getIdentifierText(key)) !== -1;
            value.inGlobalScope = _this._global;
            return cb(value, key, _this);
        });
        return _super.prototype.end.call(this, function (value, key, scope) {
            value.exported = value.exported || scope === _this
                && _this._exports !== undefined && _this._exports.indexOf(util_1.getIdentifierText(key)) !== -1;
            return cb(value, key, scope);
        });
    };
    return RootScope;
}(AbstractScope));
var NonRootScope = (function (_super) {
    tslib_1.__extends(NonRootScope, _super);
    function NonRootScope(_parent) {
        var _this = _super.call(this, false) || this;
        _this._parent = _parent;
        return _this;
    }
    NonRootScope.prototype._addUseToParent = function (use) {
        return this._parent.addUse(use, this);
    };
    return NonRootScope;
}(AbstractScope));
var EnumScope = (function (_super) {
    tslib_1.__extends(EnumScope, _super);
    function EnumScope() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    EnumScope.prototype.end = function () {
        this._applyUses();
    };
    return EnumScope;
}(NonRootScope));
var ConditionalTypeScopeState;
(function (ConditionalTypeScopeState) {
    ConditionalTypeScopeState[ConditionalTypeScopeState["Initial"] = 0] = "Initial";
    ConditionalTypeScopeState[ConditionalTypeScopeState["Extends"] = 1] = "Extends";
    ConditionalTypeScopeState[ConditionalTypeScopeState["TrueType"] = 2] = "TrueType";
    ConditionalTypeScopeState[ConditionalTypeScopeState["FalseType"] = 3] = "FalseType";
})(ConditionalTypeScopeState || (ConditionalTypeScopeState = {}));
var ConditionalTypeScope = (function (_super) {
    tslib_1.__extends(ConditionalTypeScope, _super);
    function ConditionalTypeScope() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this._state = 0;
        return _this;
    }
    ConditionalTypeScope.prototype.updateState = function (newState) {
        this._state = newState;
    };
    ConditionalTypeScope.prototype.addUse = function (use) {
        if (this._state === 2)
            return void this._uses.push(use);
        return this._parent.addUse(use, this);
    };
    return ConditionalTypeScope;
}(NonRootScope));
var FunctionScope = (function (_super) {
    tslib_1.__extends(FunctionScope, _super);
    function FunctionScope() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    FunctionScope.prototype.beginBody = function () {
        this._applyUses();
    };
    return FunctionScope;
}(NonRootScope));
var AbstractNamedExpressionScope = (function (_super) {
    tslib_1.__extends(AbstractNamedExpressionScope, _super);
    function AbstractNamedExpressionScope(_name, _domain, parent) {
        var _this = _super.call(this, parent) || this;
        _this._name = _name;
        _this._domain = _domain;
        return _this;
    }
    AbstractNamedExpressionScope.prototype.end = function (cb) {
        this._innerScope.end(cb);
        return cb({
            declarations: [this._name],
            domain: this._domain,
            exported: false,
            uses: this._uses,
            inGlobalScope: false,
        }, this._name, this);
    };
    AbstractNamedExpressionScope.prototype.addUse = function (use, source) {
        if (source !== this._innerScope)
            return this._innerScope.addUse(use);
        if (use.domain & this._domain && util_1.getIdentifierText(use.location) === util_1.getIdentifierText(this._name)) {
            this._uses.push(use);
        }
        else {
            return this._parent.addUse(use, this);
        }
    };
    AbstractNamedExpressionScope.prototype.getFunctionScope = function () {
        return this._innerScope;
    };
    AbstractNamedExpressionScope.prototype._getDestinationScope = function () {
        return this._innerScope;
    };
    return AbstractNamedExpressionScope;
}(NonRootScope));
var FunctionExpressionScope = (function (_super) {
    tslib_1.__extends(FunctionExpressionScope, _super);
    function FunctionExpressionScope(name, parent) {
        var _this = _super.call(this, name, 4, parent) || this;
        _this._innerScope = new FunctionScope(_this);
        return _this;
    }
    FunctionExpressionScope.prototype.beginBody = function () {
        return this._innerScope.beginBody();
    };
    return FunctionExpressionScope;
}(AbstractNamedExpressionScope));
var ClassExpressionScope = (function (_super) {
    tslib_1.__extends(ClassExpressionScope, _super);
    function ClassExpressionScope(name, parent) {
        var _this = _super.call(this, name, 4 | 2, parent) || this;
        _this._innerScope = new NonRootScope(_this);
        return _this;
    }
    return ClassExpressionScope;
}(AbstractNamedExpressionScope));
var BlockScope = (function (_super) {
    tslib_1.__extends(BlockScope, _super);
    function BlockScope(_functionScope, parent) {
        var _this = _super.call(this, parent) || this;
        _this._functionScope = _functionScope;
        return _this;
    }
    BlockScope.prototype.getFunctionScope = function () {
        return this._functionScope;
    };
    BlockScope.prototype._getDestinationScope = function (blockScoped) {
        return blockScoped ? this : this._functionScope;
    };
    return BlockScope;
}(NonRootScope));
function mapDeclaration(declaration) {
    return {
        declaration: declaration,
        exported: true,
        domain: getDeclarationDomain(declaration),
    };
}
var NamespaceScope = (function (_super) {
    tslib_1.__extends(NamespaceScope, _super);
    function NamespaceScope(_ambient, _hasExport, parent) {
        var _this = _super.call(this, parent) || this;
        _this._ambient = _ambient;
        _this._hasExport = _hasExport;
        _this._innerScope = new NonRootScope(_this);
        _this._exports = undefined;
        return _this;
    }
    NamespaceScope.prototype.finish = function (cb) {
        return _super.prototype.end.call(this, cb);
    };
    NamespaceScope.prototype.end = function (cb) {
        var _this = this;
        this._innerScope.end(function (variable, key, scope) {
            if (scope !== _this._innerScope ||
                !variable.exported && (!_this._ambient || _this._exports !== undefined && !_this._exports.has(util_1.getIdentifierText(key))))
                return cb(variable, key, scope);
            var namespaceVar = _this._variables.get(util_1.getIdentifierText(key));
            if (namespaceVar === undefined) {
                _this._variables.set(util_1.getIdentifierText(key), {
                    declarations: variable.declarations.map(mapDeclaration),
                    domain: variable.domain,
                    uses: variable.uses.slice(),
                });
            }
            else {
                outer: for (var _i = 0, _a = variable.declarations; _i < _a.length; _i++) {
                    var declaration = _a[_i];
                    for (var _b = 0, _c = namespaceVar.declarations; _b < _c.length; _b++) {
                        var existing = _c[_b];
                        if (existing.declaration === declaration)
                            continue outer;
                    }
                    namespaceVar.declarations.push(mapDeclaration(declaration));
                }
                namespaceVar.domain |= variable.domain;
                for (var _d = 0, _e = variable.uses; _d < _e.length; _d++) {
                    var use = _e[_d];
                    if (namespaceVar.uses.indexOf(use) !== -1)
                        continue;
                    namespaceVar.uses.push(use);
                }
            }
        });
        this._applyUses();
        this._innerScope = new NonRootScope(this);
    };
    NamespaceScope.prototype.createOrReuseNamespaceScope = function (name, exported, ambient, hasExportStatement) {
        if (!exported && (!this._ambient || this._hasExport))
            return this._innerScope.createOrReuseNamespaceScope(name, exported, ambient || this._ambient, hasExportStatement);
        return _super.prototype.createOrReuseNamespaceScope.call(this, name, exported, ambient || this._ambient, hasExportStatement);
    };
    NamespaceScope.prototype.createOrReuseEnumScope = function (name, exported) {
        if (!exported && (!this._ambient || this._hasExport))
            return this._innerScope.createOrReuseEnumScope(name, exported);
        return _super.prototype.createOrReuseEnumScope.call(this, name, exported);
    };
    NamespaceScope.prototype.addUse = function (use, source) {
        if (source !== this._innerScope)
            return this._innerScope.addUse(use);
        this._uses.push(use);
    };
    NamespaceScope.prototype.refresh = function (ambient, hasExport) {
        this._ambient = ambient;
        this._hasExport = hasExport;
    };
    NamespaceScope.prototype.markExported = function (name, _as) {
        if (this._exports === undefined)
            this._exports = new Set();
        this._exports.add(util_1.getIdentifierText(name));
    };
    NamespaceScope.prototype._getDestinationScope = function () {
        return this._innerScope;
    };
    return NamespaceScope;
}(NonRootScope));
function getEntityNameParent(name) {
    var parent = name.parent;
    while (parent.kind === ts.SyntaxKind.QualifiedName)
        parent = parent.parent;
    return parent;
}
var UsageWalker = (function () {
    function UsageWalker() {
        this._result = new Map();
    }
    UsageWalker.prototype.getUsage = function (sourceFile) {
        var _this = this;
        var variableCallback = function (variable, key) {
            _this._result.set(key, variable);
        };
        var isModule = ts.isExternalModule(sourceFile);
        this._scope = new RootScope(sourceFile.isDeclarationFile && isModule && !containsExportStatement(sourceFile), !isModule);
        var cb = function (node) {
            if (util_1.isBlockScopeBoundary(node))
                return continueWithScope(node, new BlockScope(_this._scope.getFunctionScope(), _this._scope), handleBlockScope);
            switch (node.kind) {
                case ts.SyntaxKind.ClassExpression:
                    return continueWithScope(node, node.name !== undefined
                        ? new ClassExpressionScope(node.name, _this._scope)
                        : new NonRootScope(_this._scope));
                case ts.SyntaxKind.ClassDeclaration:
                    _this._handleDeclaration(node, true, 4 | 2);
                    return continueWithScope(node, new NonRootScope(_this._scope));
                case ts.SyntaxKind.InterfaceDeclaration:
                case ts.SyntaxKind.TypeAliasDeclaration:
                    _this._handleDeclaration(node, true, 2);
                    return continueWithScope(node, new NonRootScope(_this._scope));
                case ts.SyntaxKind.EnumDeclaration:
                    _this._handleDeclaration(node, true, 7);
                    return continueWithScope(node, _this._scope.createOrReuseEnumScope(util_1.getIdentifierText(node.name), util_1.hasModifier(node.modifiers, ts.SyntaxKind.ExportKeyword)));
                case ts.SyntaxKind.ModuleDeclaration:
                    return _this._handleModule(node, continueWithScope);
                case ts.SyntaxKind.MappedType:
                    return continueWithScope(node, new NonRootScope(_this._scope));
                case ts.SyntaxKind.FunctionExpression:
                case ts.SyntaxKind.ArrowFunction:
                case ts.SyntaxKind.Constructor:
                case ts.SyntaxKind.MethodDeclaration:
                case ts.SyntaxKind.FunctionDeclaration:
                case ts.SyntaxKind.GetAccessor:
                case ts.SyntaxKind.SetAccessor:
                case ts.SyntaxKind.MethodSignature:
                case ts.SyntaxKind.CallSignature:
                case ts.SyntaxKind.ConstructSignature:
                case ts.SyntaxKind.ConstructorType:
                case ts.SyntaxKind.FunctionType:
                    return _this._handleFunctionLikeDeclaration(node, cb, variableCallback);
                case ts.SyntaxKind.ConditionalType:
                    return _this._handleConditionalType(node, cb, variableCallback);
                case ts.SyntaxKind.VariableDeclarationList:
                    _this._handleVariableDeclaration(node);
                    break;
                case ts.SyntaxKind.Parameter:
                    if (node.parent.kind !== ts.SyntaxKind.IndexSignature &&
                        (node.name.kind !== ts.SyntaxKind.Identifier ||
                            node.name.originalKeywordKind !== ts.SyntaxKind.ThisKeyword))
                        _this._handleBindingName(node.name, false, false);
                    break;
                case ts.SyntaxKind.EnumMember:
                    _this._scope.addVariable(util_1.getPropertyName(node.name), node.name, false, true, 4);
                    break;
                case ts.SyntaxKind.ImportClause:
                case ts.SyntaxKind.ImportSpecifier:
                case ts.SyntaxKind.NamespaceImport:
                case ts.SyntaxKind.ImportEqualsDeclaration:
                    _this._handleDeclaration(node, false, 7 | 8);
                    break;
                case ts.SyntaxKind.TypeParameter:
                    _this._scope.addVariable(util_1.getIdentifierText(node.name), node.name, true, false, 2);
                    break;
                case ts.SyntaxKind.ExportSpecifier:
                    if (node.propertyName !== undefined)
                        return _this._scope.markExported(node.propertyName, node.name);
                    return _this._scope.markExported(node.name);
                case ts.SyntaxKind.ExportAssignment:
                    if (node.expression.kind === ts.SyntaxKind.Identifier)
                        return _this._scope.markExported(node.expression);
                    break;
                case ts.SyntaxKind.Identifier:
                    var domain = getUsageDomain(node);
                    if (domain !== undefined)
                        _this._scope.addUse({ domain: domain, location: node });
                    return;
            }
            return ts.forEachChild(node, cb);
        };
        var continueWithScope = function (node, scope, next) {
            if (next === void 0) { next = forEachChild; }
            var savedScope = _this._scope;
            _this._scope = scope;
            next(node);
            _this._scope.end(variableCallback);
            _this._scope = savedScope;
        };
        var handleBlockScope = function (node) {
            if (node.kind === ts.SyntaxKind.CatchClause && node.variableDeclaration !== undefined)
                _this._handleBindingName(node.variableDeclaration.name, true, false);
            return ts.forEachChild(node, cb);
        };
        ts.forEachChild(sourceFile, cb);
        this._scope.end(variableCallback);
        return this._result;
        function forEachChild(node) {
            return ts.forEachChild(node, cb);
        }
    };
    UsageWalker.prototype._handleConditionalType = function (node, cb, varCb) {
        var savedScope = this._scope;
        var scope = this._scope = new ConditionalTypeScope(savedScope);
        cb(node.checkType);
        scope.updateState(1);
        cb(node.extendsType);
        scope.updateState(2);
        cb(node.trueType);
        scope.updateState(3);
        cb(node.falseType);
        scope.end(varCb);
        this._scope = savedScope;
    };
    UsageWalker.prototype._handleFunctionLikeDeclaration = function (node, cb, varCb) {
        if (node.decorators !== undefined)
            node.decorators.forEach(cb);
        var savedScope = this._scope;
        if (node.kind === ts.SyntaxKind.FunctionDeclaration)
            this._handleDeclaration(node, false, 4);
        var scope = this._scope = node.kind === ts.SyntaxKind.FunctionExpression && node.name !== undefined
            ? new FunctionExpressionScope(node.name, savedScope)
            : new FunctionScope(savedScope);
        if (node.name !== undefined)
            cb(node.name);
        if (node.typeParameters !== undefined)
            node.typeParameters.forEach(cb);
        node.parameters.forEach(cb);
        if (node.type !== undefined)
            cb(node.type);
        if (node.body !== undefined) {
            scope.beginBody();
            cb(node.body);
        }
        scope.end(varCb);
        this._scope = savedScope;
    };
    UsageWalker.prototype._handleModule = function (node, next) {
        if (node.flags & ts.NodeFlags.GlobalAugmentation)
            return next(node, this._scope.createOrReuseNamespaceScope('-global', false, true, false));
        if (node.name.kind === ts.SyntaxKind.Identifier) {
            var exported = isNamespaceExported(node);
            this._scope.addVariable(util_1.getIdentifierText(node.name), node.name, false, exported, 1 | 4);
            var ambient = util_1.hasModifier(node.modifiers, ts.SyntaxKind.DeclareKeyword);
            return next(node, this._scope.createOrReuseNamespaceScope(util_1.getIdentifierText(node.name), exported, ambient, ambient && namespaceHasExportStatement(node)));
        }
        return next(node, this._scope.createOrReuseNamespaceScope("\"" + node.name.text + "\"", false, true, namespaceHasExportStatement(node)));
    };
    UsageWalker.prototype._handleDeclaration = function (node, blockScoped, domain) {
        if (node.name !== undefined)
            this._scope.addVariable(util_1.getIdentifierText(node.name), node.name, blockScoped, util_1.hasModifier(node.modifiers, ts.SyntaxKind.ExportKeyword), domain);
    };
    UsageWalker.prototype._handleBindingName = function (name, blockScoped, exported) {
        var _this = this;
        if (name.kind === ts.SyntaxKind.Identifier)
            return this._scope.addVariable(util_1.getIdentifierText(name), name, blockScoped, exported, 4);
        util_1.forEachDestructuringIdentifier(name, function (declaration) {
            _this._scope.addVariable(util_1.getIdentifierText(declaration.name), declaration.name, blockScoped, exported, 4);
        });
    };
    UsageWalker.prototype._handleVariableDeclaration = function (declarationList) {
        var blockScoped = util_1.isBlockScopedVariableDeclarationList(declarationList);
        var exported = declarationList.parent.kind === ts.SyntaxKind.VariableStatement &&
            util_1.hasModifier(declarationList.parent.modifiers, ts.SyntaxKind.ExportKeyword);
        for (var _i = 0, _a = declarationList.declarations; _i < _a.length; _i++) {
            var declaration = _a[_i];
            this._handleBindingName(declaration.name, blockScoped, exported);
        }
    };
    return UsageWalker;
}());
function isNamespaceExported(node) {
    return node.parent.kind === ts.SyntaxKind.ModuleDeclaration || util_1.hasModifier(node.modifiers, ts.SyntaxKind.ExportKeyword);
}
function namespaceHasExportStatement(ns) {
    if (ns.body === undefined || ns.body.kind !== ts.SyntaxKind.ModuleBlock)
        return false;
    return containsExportStatement(ns.body);
}
function containsExportStatement(block) {
    for (var _i = 0, _a = block.statements; _i < _a.length; _i++) {
        var statement = _a[_i];
        if (statement.kind === ts.SyntaxKind.ExportDeclaration || statement.kind === ts.SyntaxKind.ExportAssignment)
            return true;
    }
    return false;
}
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoidXNhZ2UuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyJ1c2FnZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiOzs7QUFBQSwrQkFPZ0I7QUFDaEIsK0JBQWlDO0FBMkJqQyxJQUFrQixpQkFNakI7QUFORCxXQUFrQixpQkFBaUI7SUFDL0IsbUVBQWEsQ0FBQTtJQUNiLHlEQUFRLENBQUE7SUFDUiwyREFBUyxDQUFBO0lBQ1QsNkRBQVUsQ0FBQTtJQUNWLHVEQUE4QixDQUFBO0FBQ2xDLENBQUMsRUFOaUIsaUJBQWlCLEdBQWpCLHlCQUFpQixLQUFqQix5QkFBaUIsUUFNbEM7QUFFRCxJQUFrQixXQU9qQjtBQVBELFdBQWtCLFdBQVc7SUFDekIsdURBQWEsQ0FBQTtJQUNiLDZDQUFRLENBQUE7SUFDUiwrQ0FBUyxDQUFBO0lBQ1QscUVBQW9DLENBQUE7SUFDcEMsMkNBQThCLENBQUE7SUFDOUIsdURBQWEsQ0FBQTtBQUNqQixDQUFDLEVBUGlCLFdBQVcsR0FBWCxtQkFBVyxLQUFYLG1CQUFXLFFBTzVCO0FBRUQsU0FBZ0IsY0FBYyxDQUFDLElBQW1CO0lBQzlDLElBQU0sTUFBTSxHQUFHLElBQUksQ0FBQyxNQUFPLENBQUM7SUFDNUIsUUFBUSxNQUFNLENBQUMsSUFBSSxFQUFFO1FBQ2pCLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxhQUFhO1lBQzVCLFNBQXdCO1FBQzVCLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQywyQkFBMkI7WUFDMUMsT0FBMkIsTUFBTSxDQUFDLE1BQU8sQ0FBQyxLQUFLLEtBQUssRUFBRSxDQUFDLFVBQVUsQ0FBQyxpQkFBaUI7Z0JBQy9FLE1BQU0sQ0FBQyxNQUFPLENBQUMsTUFBTyxDQUFDLElBQUksS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLG9CQUFvQjtnQkFDbEUsQ0FBQztnQkFDRCxDQUFDLEVBQWtCLENBQUM7UUFDNUIsS0FBSyxFQUFFLENBQUMsVUFBVSxDQUFDLFNBQVM7WUFDeEIsT0FBTyxLQUFvRCxDQUFD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