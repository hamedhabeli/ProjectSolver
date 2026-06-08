from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

from psai.logic.orchestrator import run_cycle, run_workflow
from psai.llm.provider import LLMProvider
from psai.state_store import StateStore

JsonDict = dict[str, Any]


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def _rpc_result(id_: Any, result: Any) -> JsonDict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _rpc_error(id_: Any, code: int, message: str, data: Any | None = None) -> JsonDict:
    err: JsonDict = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def _require(params: JsonDict, key: str, typ: type) -> Any:
    if key not in params:
        raise JsonRpcError(-32602, f"Missing param: {key}")
    val = params[key]
    if not isinstance(val, typ):
        raise JsonRpcError(-32602, f"Invalid param type for {key}: expected {typ.__name__}")
    return val


class _NoopLLM(LLMProvider):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM not configured for this server instance")


class ApiServer:
    """
    JSON-RPC 2.0 server over stdin/stdout.

    workflow.run is the one-shot integration route that executes:
      formalize -> check_consistency -> explain_contradiction
    """

    def __init__(self, store: StateStore, llm: Optional[LLMProvider] = None) -> None:
        self.store = store
        self.llm = llm or _NoopLLM()

        self._stdout_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)

        self._routes: dict[str, Callable[[JsonDict], Any]] = {
            "repo.create": self._repo_create,
            "repo.status": self._repo_status,
            "commit": self._commit,
            "checkout": self._checkout,
            "log": self._log,
            "cycle.run": self._cycle_run,
            "workflow.run": self._workflow_run,
        }

    def serve_forever(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            self._executor.submit(self._handle_line, line)

    def _write(self, obj: JsonDict) -> None:
        data = json.dumps(obj, ensure_ascii=False)
        with self._stdout_lock:
            sys.stdout.write(data + "\n")
            sys.stdout.flush()

    def _handle_line(self, line: str) -> None:
        try:
            req = json.loads(line)
            if isinstance(req, list):
                self._write(_rpc_error(None, -32600, "Batch requests not supported"))
                return

            if not isinstance(req, dict):
                self._write(_rpc_error(None, -32600, "Invalid Request"))
                return

            if req.get("jsonrpc") != "2.0":
                self._write(_rpc_error(req.get("id"), -32600, "Invalid Request: jsonrpc must be '2.0'"))
                return

            method = req.get("method")
            id_ = req.get("id")
            params = req.get("params", {})
            if not isinstance(method, str):
                self._write(_rpc_error(id_, -32600, "Invalid Request: method must be string"))
                return
            if not isinstance(params, dict):
                self._write(_rpc_error(id_, -32602, "Invalid params: params must be object"))
                return

            handler = self._routes.get(method)
            if handler is None:
                self._write(_rpc_error(id_, -32601, f"Method not found: {method}"))
                return

            result = handler(params)
            self._write(_rpc_result(id_, result))

        except JsonRpcError as e:
            self._write(_rpc_error(None, e.code, e.message, e.data))
        except json.JSONDecodeError as e:
            self._write(_rpc_error(None, -32700, "Parse error", {"detail": str(e)}))
        except Exception as e:  # noqa: BLE001
            self._write(_rpc_error(None, -32603, "Internal error", {"detail": str(e)}))

    def _repo_create(self, params: JsonDict) -> JsonDict:
        title = _require(params, "title", str)
        initial_payload = params.get("initial_payload", {})
        if not isinstance(initial_payload, dict):
            raise JsonRpcError(-32602, "Invalid param type for initial_payload: expected object")

        status = self.store.repo_create(title=title, initial_payload=initial_payload)
        return asdict(status)

    def _repo_status(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        status = self.store.repo_status(repo_id)
        return asdict(status)

    def _commit(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        parent_id = _require(params, "parent_id", str)
        message = _require(params, "message", str)
        payload = _require(params, "payload", dict)

        node = self.store.commit(repo_id=repo_id, parent_id=parent_id, message=message, payload=payload)
        return asdict(node)

    def _checkout(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        node_id = _require(params, "node_id", str)

        node = self.store.checkout(repo_id=repo_id, node_id=node_id)
        return asdict(node)

    def _log(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        from_node_id = params.get("from_node_id", None)
        if from_node_id is not None and not isinstance(from_node_id, str):
            raise JsonRpcError(-32602, "Invalid param type for from_node_id: expected string or null")

        limit = params.get("limit", 50)
        if not isinstance(limit, int) or limit < 0:
            raise JsonRpcError(-32602, "Invalid param type for limit: expected non-negative integer")

        nodes = self.store.log(repo_id=repo_id, from_node_id=from_node_id, limit=limit)
        return {"nodes": [asdict(n) for n in nodes]}

    def _cycle_run(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        workspace_id = _require(params, "workspace_id", str)
        step = _require(params, "step", str)

        res = run_cycle(repo_id=repo_id, workspace_id=workspace_id, step=step, store=self.store, llm=self.llm)
        return res

    def _workflow_run(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        workspace_id = _require(params, "workspace_id", str)

        res = run_workflow(repo_id=repo_id, workspace_id=workspace_id, store=self.store, llm=self.llm)
        return res


def main() -> None:
    db_path = Path("psai.sqlite3")
    store = StateStore(db_path=db_path)
    server = ApiServer(store=store)
    server.serve_forever()


if __name__ == "__main__":
    main()
