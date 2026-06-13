from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import psai.logic.orchestrator as orchestrator_module
from psai.llm.provider import GeminiModelInfo, GeminiProvider, LLMProvider, list_gemini_models
from psai.logic.json_utils import llm_json_with_retry
from psai.logic.orchestrator import run_cycle, run_workflow
from psai.logic.syntax_gate import validate_smtlib2
from psai.solvers.mus import extract_mus
from psai.solvers.z3_backend import check_sat, prove_goal
from psai.state_store import StateStore

# Patch the orchestration helper in one place so the desktop app can accept
# Gemini outputs that are fenced or wrapped with lightweight prose.
orchestrator_module._llm_json_with_retry = llm_json_with_retry

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


def _optional_str(params: JsonDict, key: str) -> Optional[str]:
    if key not in params:
        return None
    val = params[key]
    if val is None:
        return None
    if not isinstance(val, str):
        raise JsonRpcError(-32602, f"Invalid param type for {key}: expected string or null")
    return val


def _optional_int(params: JsonDict, key: str, default: int) -> int:
    val = params.get(key, default)
    if not isinstance(val, int):
        raise JsonRpcError(-32602, f"Invalid param type for {key}: expected integer")
    return val


def _optional_float(params: JsonDict, key: str, default: float) -> float:
    val = params.get(key, default)
    if isinstance(val, int):
        return float(val)
    if not isinstance(val, float):
        raise JsonRpcError(-32602, f"Invalid param type for {key}: expected number")
    return val


class _NoopLLM(LLMProvider):
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("LLM not configured for this server instance")


class ApiServer:
    """JSON-RPC 2.0 server over stdin/stdout."""

    def __init__(self, store: StateStore, llm: Optional[LLMProvider] = None) -> None:
        self.store = store
        self._override_llm = llm
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
            "llm.get_config": self._llm_get_config,
            "llm.set_config": self._llm_set_config,
            "llm.gemini.list_models": self._llm_gemini_list_models,
            "llm.gemini.test_key": self._llm_gemini_test_key,
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
            detail = str(e).strip()
            message = f"Internal error: {detail}" if detail else "Internal error"
            self._write(
                _rpc_error(
                    None,
                    -32603,
                    message,
                    {"detail": detail, "exception": type(e).__name__},
                )
            )

    def _runtime_llm(self) -> LLMProvider:
        if self._override_llm is not None:
            return self._override_llm

        config = self.store.get_llm_config()
        provider = str(config.get("provider", "gemini") or "gemini")
        if provider != "gemini":
            return _NoopLLM()

        api_key = str(config.get("api_key", "") or "")
        model = str(config.get("model", "") or "")
        if not api_key or not model:
            return _NoopLLM()

        try:
            temperature = float(config.get("temperature", 0.2))
            top_p = float(config.get("top_p", 0.95))
            timeout_s = int(config.get("timeout_s", 30))
            max_retries = int(config.get("max_retries", 2))
        except Exception:  # noqa: BLE001
            return _NoopLLM()

        return GeminiProvider(
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            max_retries=max_retries,
        )

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
        from_node_id = _optional_str(params, "from_node_id")
        limit = _optional_int(params, "limit", 50)
        nodes = self.store.log(repo_id=repo_id, from_node_id=from_node_id, limit=limit)
        return {"nodes": [asdict(n) for n in nodes]}

    def _cycle_run(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        workspace_id = _require(params, "workspace_id", str)
        step = _require(params, "step", str)
        res = run_cycle(repo_id=repo_id, workspace_id=workspace_id, step=step, store=self.store, llm=self._runtime_llm())
        return res

    def _workflow_run(self, params: JsonDict) -> JsonDict:
        repo_id = _require(params, "repo_id", str)
        workspace_id = _require(params, "workspace_id", str)
        res = run_workflow(repo_id=repo_id, workspace_id=workspace_id, store=self.store, llm=self._runtime_llm())
        return res

    def _llm_get_config(self, params: JsonDict) -> JsonDict:
        _ = params
        return self.store.get_llm_config()

    def _llm_set_config(self, params: JsonDict) -> JsonDict:
        config = params.get("config", params)
        if not isinstance(config, dict):
            raise JsonRpcError(-32602, "config must be an object")
        provider = str(config.get("provider", "gemini") or "gemini")
        if provider != "gemini":
            raise JsonRpcError(-32602, "Only provider='gemini' is supported in this build")
        normalized = {
            "provider": provider,
            "api_key": str(config.get("api_key", "") or ""),
            "model": str(config.get("model", "") or ""),
            "temperature": _optional_float(config, "temperature", 0.2),
            "top_p": _optional_float(config, "top_p", 0.95),
            "timeout_s": _optional_int(config, "timeout_s", 30),
            "max_retries": _optional_int(config, "max_retries", 2),
        }
        self.store.set_llm_config(normalized)
        return self.store.get_llm_config()

    def _llm_gemini_list_models(self, params: JsonDict) -> JsonDict:
        api_key = _optional_str(params, "api_key")
        if not api_key:
            api_key = str(self.store.get_llm_config().get("api_key", "") or "")
        if not api_key:
            raise JsonRpcError(-32602, "Missing param: api_key")
        timeout_s = _optional_int(params, "timeout_s", int(self.store.get_llm_config().get("timeout_s", 30)))
        models = list_gemini_models(api_key=api_key, timeout_s=timeout_s)
        return {"models": [self._model_to_dict(m) for m in models]}

    def _llm_gemini_test_key(self, params: JsonDict) -> JsonDict:
        api_key = _optional_str(params, "api_key")
        if not api_key:
            api_key = str(self.store.get_llm_config().get("api_key", "") or "")
        if not api_key:
            raise JsonRpcError(-32602, "Missing param: api_key")
        timeout_s = _optional_int(params, "timeout_s", int(self.store.get_llm_config().get("timeout_s", 30)))
        models = list_gemini_models(api_key=api_key, timeout_s=timeout_s)
        return {"ok": True, "models_count": len(models), "models": [self._model_to_dict(m) for m in models]}

    def _model_to_dict(self, model: Any) -> JsonDict:
        if is_dataclass(model):
            return asdict(model)
        if isinstance(model, dict):
            return model
        raise JsonRpcError(-32603, "Invalid model object returned from Gemini model list")


def main() -> None:
    db_path_env = os.environ.get("PROJECTSOLVER_DB_PATH")
    if db_path_env:
        db_path = Path(db_path_env)
    else:
        app_dir = Path.home() / ".projectsolver"
        app_dir.mkdir(parents=True, exist_ok=True)
        db_path = app_dir / "psai.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = StateStore(db_path=db_path)
    server = ApiServer(store=store)
    server.serve_forever()


if __name__ == "__main__":
    main()
