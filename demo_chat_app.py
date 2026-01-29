"""
app.py
A minimal chat-group app + Vanna FastAPI server in one file.

Features (core):
- Multi-user accounts (simple REST create/list)
- Group chats
- Send messages (text + optional image_url) with timestamp stored in DB
- Query chat history by group, by user, by time range
- Vanna Agent server (Text-to-SQL via RunSqlTool) connected to the same SQLite DB

Run:
  pip install fastapi uvicorn vanna
  export DEEPSEEK_API_KEY="your_key"
  python app.py

Then:
  - REST docs: http://127.0.0.1:8000/docs
  - Vanna endpoints are mounted under /vanna (same server)
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---- Vanna imports (based on your snippet) ----
from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.integrations.sqlite import SqliteRunner
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool
from vanna.integrations.local.agent_memory import DemoAgentMemory

# If you want DeepSeek via OpenAI-compatible API:
from vanna.integrations.openai import OpenAILlmService

from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


DB_PATH = "./demo_chat_app.sqlite"

load_dotenv()


# =========================================================
# 1) Database init (creates empty DB file + schema)
# =========================================================
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS groups (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS group_members (
  group_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  role TEXT NOT NULL DEFAULT 'member',  -- e.g. member/admin
  joined_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (group_id, user_id),
  FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  group_id INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  content TEXT,            -- nullable if only image_url
  image_url TEXT,          -- optional
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_group_time ON messages(group_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, created_at);
"""


def init_db(db_path: str = DB_PATH) -> None:
    """
    Creating an 'empty' SQLite DB is equivalent to creating/opening the file.
    Then we apply schema. Safe to call on every startup.
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# =========================================================
# 2) REST API models
# =========================================================
class CreateUserReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)


class CreateGroupReq(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)


class AddMemberReq(BaseModel):
    username: str
    role: str = "member"


class SendMessageReq(BaseModel):
    username: str
    content: Optional[str] = None
    image_url: Optional[str] = None


class MessageOut(BaseModel):
    id: int
    group_id: int
    username: str
    content: Optional[str]
    image_url: Optional[str]
    created_at: str


# =========================================================
# 3) REST helpers
# =========================================================
def get_user_id(conn: sqlite3.Connection, username: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"User not found: {username}")
    return int(row["id"])


def get_group_id(conn: sqlite3.Connection, group_name: str) -> int:
    row = conn.execute("SELECT id FROM groups WHERE name = ?", (group_name,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Group not found: {group_name}")
    return int(row["id"])


# =========================================================
# 4) Vanna user resolver (cookie-based like your code)
# =========================================================
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie("vanna_email")
        if not user_email:
            raise ValueError("Missing 'vanna_email' cookie for user identification")

        if user_email == "admin@example.com":
            return User(id="admin1", email=user_email, group_memberships=["admin"])

        return User(id="user1", email=user_email, group_memberships=["user"])

# =========================================================
# 5) Build FastAPI + mount Vanna server
# =========================================================
init_db(DB_PATH)

app = FastAPI(title="Chat Group App + Vanna", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Build Vanna Agent (same DB) ----
tools = ToolRegistry()
tools.register_local_tool(
    RunSqlTool(sql_runner=SqliteRunner(database_path=DB_PATH)),
    access_groups=["admin", "user"],
)
tools.register_local_tool(VisualizeDataTool(), access_groups=["admin", "user"])

agent_memory = DemoAgentMemory(max_items=1000)
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=["admin"])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=["admin", "user"])

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
if not DEEPSEEK_API_KEY:
    # You can still run REST APIs without LLM, but Vanna needs the key to work properly.
    # We'll not crash; we'll warn in logs.
    print("[WARN] DEEPSEEK_API_KEY is empty. Vanna LLM calls may fail.")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "").strip()
if not DEEPSEEK_BASE_URL:
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "").strip()
if not DEEPSEEK_MODEL:
    DEEPSEEK_MODEL = "deepseek-chat"

llm = OpenAILlmService(
    api_key=DEEPSEEK_API_KEY,
    model=DEEPSEEK_MODEL,
    base_url=DEEPSEEK_BASE_URL,
)

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=SimpleUserResolver(),
    config=AgentConfig(max_tool_iterations=50),
    agent_memory=agent_memory,
)

from vanna.servers.fastapi.routes import register_chat_routes
from vanna.servers.base import ChatHandler

chat_handler = ChatHandler(agent)

# 默认会注册类似：
# - POST /api/vanna/v2/chat_sse
# - GET  / (可选 web UI，看版本实现)
register_chat_routes(app, chat_handler)


# =========================================================
# 6) REST endpoints for chat app
# =========================================================
@app.get("/health")
def health():
    return {"ok": True, "db_path": DB_PATH}


@app.post("/users")
def create_user(req: CreateUserReq):
    conn = get_conn()
    try:
        try:
            conn.execute("INSERT INTO users (username) VALUES (?)", (req.username,))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Username already exists")
        row = conn.execute("SELECT id, username, created_at FROM users WHERE username = ?", (req.username,)).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.get("/users")
def list_users():
    conn = get_conn()
    try:
        rows = conn.execute("SELECT id, username, created_at FROM users ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/groups")
def create_group(req: CreateGroupReq):
    conn = get_conn()
    try:
        try:
            conn.execute("INSERT INTO groups (name) VALUES (?)", (req.name,))
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Group name already exists")
        row = conn.execute("SELECT id, name, created_at FROM groups WHERE name = ?", (req.name,)).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.get("/groups")
def list_groups():
    conn = get_conn()
    try:
        rows = conn.execute("SELECT id, name, created_at FROM groups ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


@app.post("/groups/{group_name}/members")
def add_member(group_name: str, req: AddMemberReq):
    conn = get_conn()
    try:
        gid = get_group_id(conn, group_name)
        uid = get_user_id(conn, req.username)
        try:
            conn.execute(
                "INSERT INTO group_members (group_id, user_id, role) VALUES (?, ?, ?)",
                (gid, uid, req.role),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="User is already a member of this group")

        row = conn.execute(
            """
            SELECT g.name AS group_name, u.username, gm.role, gm.joined_at
            FROM group_members gm
            JOIN users u ON u.id = gm.user_id
            JOIN groups g ON g.id = gm.group_id
            WHERE gm.group_id = ? AND gm.user_id = ?
            """,
            (gid, uid),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.post("/groups/{group_name}/messages")
def send_message(group_name: str, req: SendMessageReq):
    if (req.content is None or req.content.strip() == "") and (req.image_url is None or req.image_url.strip() == ""):
        raise HTTPException(status_code=400, detail="Either content or image_url must be provided")

    conn = get_conn()
    try:
        gid = get_group_id(conn, group_name)
        uid = get_user_id(conn, req.username)

        # ensure membership
        mem = conn.execute(
            "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
            (gid, uid),
        ).fetchone()
        if not mem:
            raise HTTPException(status_code=403, detail="User is not a member of this group")

        conn.execute(
            "INSERT INTO messages (group_id, user_id, content, image_url) VALUES (?, ?, ?, ?)",
            (gid, uid, req.content, req.image_url),
        )
        conn.commit()

        row = conn.execute(
            """
            SELECT m.id, m.group_id, u.username, m.content, m.image_url, m.created_at
            FROM messages m
            JOIN users u ON u.id = m.user_id
            WHERE m.rowid = last_insert_rowid()
            """
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


@app.get("/groups/{group_name}/messages", response_model=List[MessageOut])
def get_messages(
    group_name: str,
    limit: int = 50,
    before: Optional[str] = None,  # ISO-like text, e.g. "2026-01-18 12:00:00"
    after: Optional[str] = None,
):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 500")

    conn = get_conn()
    try:
        gid = get_group_id(conn, group_name)

        where = ["m.group_id = ?"]
        params = [gid]

        if after:
            where.append("m.created_at >= ?")
            params.append(after)
        if before:
            where.append("m.created_at < ?")
            params.append(before)

        where_sql = " AND ".join(where)

        rows = conn.execute(
            f"""
            SELECT m.id, m.group_id, u.username, m.content, m.image_url, m.created_at
            FROM messages m
            JOIN users u ON u.id = m.user_id
            WHERE {where_sql}
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()

        return [MessageOut(**dict(r)) for r in rows]
    finally:
        conn.close()


# =========================================================
# 7) Entrypoint
# =========================================================
if __name__ == "__main__":
    # Use uvicorn so everything runs in one server.
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
