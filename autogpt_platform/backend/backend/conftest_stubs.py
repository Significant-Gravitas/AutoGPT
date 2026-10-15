import sys, types
from enum import Enum
from pydantic import BaseModel, ConfigDict
import prisma
import prisma as _prisma_pkg
prisma.Json = dict
import datetime as _datetime
import uuid as _uuid
class _FakeRecord:
    def __init__(self, data=None):
        self._data = dict(data or {})
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self._data:
            return self._data[name]
        if name == 'id':
            return str(_uuid.uuid4())
        if name == 'email':
            return "default@example.com"
        if name in ('created_at', 'updated_at', 'createdAt', 'updatedAt'):
            return _datetime.datetime.now(_datetime.timezone.utc)
        if name in ('emailVerified',):
            return True
        if name in ('integrations',):
            return ""
        if name in ('maxEmailsPerDay',):
            return 3
        return None
class _ModelProxy:
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        async def _noop(*a, **kw):
            if name in ('create', 'update', 'upsert'):
                return _FakeRecord(kw.get('data') or {})
            if name in ('find_unique', 'find_first', 'find_unique_or_raise', 'find_many'):
                return _FakeRecord()
            return None
        return _noop
class Prisma:
    def __init__(self, *a, **kw):
        self._models = {}
    def __getstate__(self): return {}
    def __setstate__(self, state): pass
    def __reduce__(self): return (_make_prisma, ())
    async def connect(self, *a, **kw): pass
    async def disconnect(self, *a, **kw): pass
    def is_connected(self, *a, **kw): return True
    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        p = self._models.get(name)
        if p is None:
            p = _ModelProxy()
            self._models[name] = p
        return p
def _make_prisma():
    return Prisma()
prisma.Prisma = Prisma
prisma.Client = Prisma
def _register_stub_modules():
    import sys as _sys
    _sys.modules.setdefault('prisma.models', models)
    _sys.modules.setdefault('prisma.types', types_mod)
    _sys.modules.setdefault('prisma.enums', enums)
    _sys.modules.setdefault('prisma.errors', errors)
prisma.register_stub_modules = _register_stub_modules
models = types.ModuleType('prisma.models')
class _PrismaClient:
    def __getattr__(self, name):
        async def _noop(*a, **kw):
            return None
        return _noop
def _model(name):
    def _get_pydantic_core_schema(self_type, handler):
        from pydantic import ConfigDict
        from pydantic.main import BaseModel
        return handler(BaseModel)
    class _M(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        prisma = staticmethod(lambda: _PrismaClient())
        def __init__(self, *a, **kw):
            super().__init__()
            for k, v in kw.items():
                setattr(self, k, v)
    _M.__name__ = name
    _M.__get_pydantic_core_schema__ = _get_pydantic_core_schema
    return _M
_MODEL_NAMES = ('Agent', 'AgentVersion', 'AgentBlock', 'AgentBlockVersion', 'AgentGraph', 'AgentNode', 'AgentNodeExecution', 'Execution', 'ExecutionBlock', 'ExecutionContext', 'BlockInput', 'BlockOutput', 'BlockOutputEntry', 'BlockInputEntry', 'Credential', 'APIKey', 'User', 'Workspace', 'UserWorkspace', 'UserWorkspaceFile', 'CreditTransaction', 'UserBalance', 'CreditRefundRequest', 'Subscription', 'Notification', 'NotificationEvent', 'NotificationRecipient', 'Schedule', 'ScheduleTrigger', 'StoreSubmission', 'StoreReview', 'OnboardingSession', 'UserOnboarding', 'ChatMessage', 'ChatSession', 'CoPilotUnderstanding', 'OAuthAccessToken', 'OAuthApplication', 'OAuthAuthorizationCode', 'OAuthRefreshToken', 'PendingHumanReview', 'IntegrationWebhook', 'AgentPreset')
for _n in _MODEL_NAMES:
    if not hasattr(models, _n):
        setattr(models, _n, _model(_n))
models.mv_suggested_blocks = _model('mv_suggested_blocks')
class _AnyModel:
    prisma = staticmethod(lambda: None)
    def __init_subclass__(cls, **kw): super().__init_subclass__(**kw)
models.__getattr__ = lambda name: _model(name)
types_mod = types.ModuleType('prisma.types')
class _Input(dict): pass
_INPUT_NAMES = ('AgentBlockCreateInput', 'CreditRefundRequestCreateInput', 'CreditTransactionWhereInput', 'CreditTransactionCreateInput', 'UserWhereInput', 'UserCreateInput', 'UserUpdateInput', 'UserBalanceWhereInput', 'UserOnboardingCreateInput', 'UserOnboardingUpdateInput', 'UserWorkspaceFileWhereInput', 'AgentWhereInput', 'AgentVersionWhereInput', 'AgentBlockVersionWhereInput', 'ExecutionWhereInput', 'ExecutionBlockWhereInput', 'BlockInputWhereInput', 'BlockOutputWhereInput', 'CredentialWhereInput', 'APIKeyWhereInput', 'APIKeyWhereUniqueInput', 'WorkspaceWhereInput', 'UserWorkspaceWhereInput', 'SubscriptionWhereInput', 'NotificationWhereInput', 'NotificationEventWhereInput', 'ScheduleWhereInput', 'StoreSubmissionWhereInput', 'StoreReviewWhereInput', 'OnboardingSessionWhereInput', 'ChatMessageWhereInput', 'ChatSessionWhereInput', 'CoPilotUnderstandingWhereInput', 'OAuthApplicationUpdateInput', 'PendingHumanReviewUpdateInput', 'ProfileCreateInput', 'UserNotificationBatch')
for _n in _INPUT_NAMES:
    if not hasattr(types_mod, _n):
        setattr(types_mod, _n, _Input)
types_mod.JsonFilter = _Input
class Serializable: pass
types_mod.Serializable = Serializable
types_mod.__getattr__ = lambda name: _Input
enums = types.ModuleType('prisma.enums')
class AgentExecutionStatus(str, Enum): INCOMPLETE='INCOMPLETE'; QUEUED='QUEUED'; RUNNING='RUNNING'; COMPLETED='COMPLETED'; TERMINATED='TERMINATED'; FAILED='FAILED'; REVIEW='REVIEW'
class APIKeyPermission(str, Enum): IDENTITY='IDENTITY'; EXECUTE_GRAPH='EXECUTE_GRAPH'; READ_GRAPH='READ_GRAPH'; WRITE_GRAPH='WRITE_GRAPH'; EXECUTE_BLOCK='EXECUTE_BLOCK'; READ_BLOCK='READ_BLOCK'; READ_STORE='READ_STORE'; WRITE_LIBRARY='WRITE_LIBRARY'; USE_TOOLS='USE_TOOLS'; MANAGE_INTEGRATIONS='MANAGE_INTEGRATIONS'; READ_INTEGRATIONS='READ_INTEGRATIONS'; DELETE_INTEGRATIONS='DELETE_INTEGRATIONS'
class APIKeyStatus(str, Enum): ACTIVE='ACTIVE'; REVOKED='REVOKED'; SUSPENDED='SUSPENDED'
class ContentType(str, Enum): STORE_AGENT='STORE_AGENT'; BLOCK='BLOCK'; INTEGRATION='INTEGRATION'; DOCUMENTATION='DOCUMENTATION'; LIBRARY_AGENT='LIBRARY_AGENT'
class SubscriptionTier(str, Enum): NO_TIER='NO_TIER'; TRIAL='TRIAL'; BASIC='BASIC'; PRO='PRO'; MAX='MAX'; BUSINESS='BUSINESS'
class BriefingFrequency(str, Enum): DAILY='DAILY'; WEEKLY='WEEKLY'; MONTHLY='MONTHLY'; OFF='OFF'
class CreditTransactionType(str, Enum): TOP_UP='TOP_UP'; USAGE='USAGE'; GRANT='GRANT'; REFUND='REFUND'; CARD_CHECK='CARD_CHECK'
class OnboardingStep(str, Enum):
    WELCOME='WELCOME'; USAGE_REASON='USAGE_REASON'; INTEGRATIONS='INTEGRATIONS'; AGENT_CHOICE='AGENT_CHOICE'; AGENT_NEW_RUN='AGENT_NEW_RUN'; AGENT_INPUT='AGENT_INPUT'; CONGRATS='CONGRATS'; VISIT_COPILOT='VISIT_COPILOT'; GET_RESULTS='GET_RESULTS'; MARKETPLACE_VISIT='MARKETPLACE_VISIT'; MARKETPLACE_ADD_AGENT='MARKETPLACE_ADD_AGENT'; MARKETPLACE_RUN_AGENT='MARKETPLACE_RUN_AGENT'; BUILDER_SAVE_AGENT='BUILDER_SAVE_AGENT'; RE_RUN_AGENT='RE_RUN_AGENT'; SCHEDULE_AGENT='SCHEDULE_AGENT'; RUN_AGENTS='RUN_AGENTS'; RUN_3_DAYS='RUN_3_DAYS'; TRIGGER_WEBHOOK='TRIGGER_WEBHOOK'; RUN_14_DAYS='RUN_14_DAYS'; RUN_AGENTS_100='RUN_AGENTS_100'; BUILDER_OPEN='BUILDER_OPEN'; BUILDER_RUN_AGENT='BUILDER_RUN_AGENT'
class CreditRefundRequestStatus(str, Enum): PENDING='PENDING'; APPROVED='APPROVED'; REJECTED='REJECTED'
class NotificationType(str, Enum): EMAIL='EMAIL'; PUSH='PUSH'; WEBHOOK='WEBHOOK'
class ReviewStatus(str, Enum): WAITING='WAITING'; APPROVED='APPROVED'; REJECTED='REJECTED'
class SubmissionStatus(str, Enum): DRAFT='DRAFT'; PENDING='PENDING'; APPROVED='APPROVED'; REJECTED='REJECTED'
for _n in ('AgentExecutionStatus', 'APIKeyPermission', 'APIKeyStatus', 'ContentType', 'SubscriptionTier', 'BriefingFrequency', 'CreditTransactionType', 'OnboardingStep', 'CreditRefundRequestStatus', 'NotificationType', 'ReviewStatus', 'SubmissionStatus'):
    setattr(enums, _n, globals()[_n])
errors = types.ModuleType('prisma.errors')
class PrismaError(Exception): pass
class UniqueViolationError(PrismaError): pass
class DataError(PrismaError): pass
class ModelNotFoundError(PrismaError): pass
class ModelExistsError(PrismaError): pass
for _n in ('PrismaError', 'UniqueViolationError', 'DataError', 'ModelNotFoundError', 'ModelExistsError'):
    setattr(errors, _n, globals()[_n])
sys.modules['prisma.models']=models
sys.modules['prisma.types']=types_mod
sys.modules['prisma.enums']=enums
sys.modules['prisma.errors']=errors
prisma.models = models; prisma.types = types_mod; prisma.enums = enums; prisma.errors = errors
prisma.Prisma = Prisma
prisma.Client = Prisma
_register_stub_modules()
