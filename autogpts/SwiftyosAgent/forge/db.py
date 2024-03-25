import datetime
import uuid

from sqlalchemy import Column, DateTime, String
from sqlalchemy.exc import SQLAlchemyError

from .sdk import AgentDB, Base, ForgeLogger, NotFoundError

LOG = ForgeLogger(__name__)


class ChatModel(Base):
    __tablename__ = "chat"
    msg_id = Column(String, primary_key=True, index=True)
    task_id = Column(String)
    role = Column(String)
    content = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class ActionModel(Base):
    __tablename__ = "action"
    action_id = Column(String, primary_key=True, index=True)
    task_id = Column(String)
    name = Column(String)
    args = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    modified_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class ForgeDatabase(AgentDB):
    async def add_chat_history(self, task_id, messages):
        for message in messages:
            await self.add_chat_message(task_id, message["role"], message["content"])

    async def add_chat_message(self, task_id, role, content):
        if self.debug_enabled:
            LOG.debug("Creating new task")
        try:
            with self.Session() as session:
                mew_msg = ChatModel(
                    msg_id=str(uuid.uuid4()),
                    task_id=task_id,
                    role=role,
                    content=content,
                )
                session.add(mew_msg)
                session.commit()
                session.refresh(mew_msg)
                if self.debug_enabled:
                    LOG.debug(
                        f"Created new Chat message with task_id: {mew_msg.msg_id}"
                    )
                return mew_msg
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating task: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while creating task: {e}")
            raise

    async def get_chat_history(self, task_id):
        if self.debug_enabled:
            LOG.debug(f"Getting chat history with task_id: {task_id}")
        try:
            with self.Session() as session:
                if messages := (
                    session.query(ChatModel)
                    .filter(ChatModel.task_id == task_id)
                    .order_by(ChatModel.created_at)
                    .all()
                ):
                    return [{"role": m.role, "content": m.content} for m in messages]

                else:
                    LOG.error(f"Chat history not found with task_id: {task_id}")
                    raise NotFoundError("Chat history not found")
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while getting chat history: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while getting chat history: {e}")
            raise

    async def create_action(self, task_id, name, args):
        try:
            with self.Session() as session:
                new_action = ActionModel(
                    action_id=str(uuid.uuid4()),
                    task_id=task_id,
                    name=name,
                    args=str(args),
                )
                session.add(new_action)
                session.commit()
                session.refresh(new_action)
                if self.debug_enabled:
                    LOG.debug(
                        f"Created new Action with task_id: {new_action.action_id}"
                    )
                return new_action
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while creating action: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while creating action: {e}")
            raise

    async def get_action_history(self, task_id):
        if self.debug_enabled:
            LOG.debug(f"Getting action history with task_id: {task_id}")
        try:
            with self.Session() as session:
                if actions := (
                    session.query(ActionModel)
                    .filter(ActionModel.task_id == task_id)
                    .order_by(ActionModel.created_at)
                    .all()
                ):
                    return [{"name": a.name, "args": a.args} for a in actions]

                else:
                    LOG.error(f"Action history not found with task_id: {task_id}")
                    raise NotFoundError("Action history not found")
        except SQLAlchemyError as e:
            LOG.error(f"SQLAlchemy error while getting action history: {e}")
            raise
        except NotFoundError as e:
            raise
        except Exception as e:
            LOG.error(f"Unexpected error while getting action history: {e}")
            raise
