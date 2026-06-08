import { useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type RpcError = {
  code: number;
  message: string;
  data?: unknown;
};

type RpcEnvelope<T = unknown> =
  | {
      jsonrpc: '2.0';
      id: string;
      result: T;
    }
  | {
      jsonrpc: '2.0';
      id: string;
      error: RpcError;
    };

type WorkflowResult = {
  step: string;
  status: string;
  formalize?: {
    step: string;
    status: string;
    valid_items?: Array<{
      item_id: string;
      fa_text: string;
      formal_smt2: string;
    }>;
    invalid_items?: Array<{
      item_id: string;
      fa_text: string;
      formal_smt2: string;
      syntax: {
        valid: boolean;
        errors: Array<{ code: string; message: string; detail?: string }>;
      };
    }>;
  };
  check_consistency?: {
    step: string;
    sat: {
      status: 'sat' | 'unsat' | 'unknown';
      reason_unknown?: string | null;
    };
  };
  explain_contradiction?: {
    step: string;
    status: string;
    unsat_core_item_ids?: string[];
    explanation?: {
      nl_summary?: string;
      choices?: Array<{
        choice_id: string;
        action: string;
        target_item_id?: string;
        nl: string;
      }>;
    };
  } | null;
  final?: unknown;
};

function newId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function rpc<T>(method: string, params: Record<string, unknown>): Promise<RpcEnvelope<T>> {
  const request = {
    jsonrpc: '2.0',
    id: newId(),
    method,
    params,
  };

  const responseJson = await invoke<string>('rpc_call', {
    request_json: JSON.stringify(request),
  });

  return JSON.parse(responseJson) as RpcEnvelope<T>;
}

export default function App() {
  const [problemText, setProblemText] = useState(
    'پروژه الف نباید با پروژه ب همزمان فعال باشد. همچنین پروژه الف باید همزمان فعال باشد.',
  );
  const [loading, setLoading] = useState(false);
  const [rpcError, setRpcError] = useState<string | null>(null);
  const [result, setResult] = useState<WorkflowResult | null>(null);

  const summary = useMemo(() => {
    if (!result) return null;
    return result.explain_contradiction?.explanation?.nl_summary ?? null;
  }, [result]);

  const choices = useMemo(() => {
    if (!result) return [];
    return result.explain_contradiction?.explanation?.choices ?? [];
  }, [result]);

  async function handleRun() {
    setLoading(true);
    setRpcError(null);
    setResult(null);

    try {
      const createRes = await rpc<{
        repo_id: string;
        head: string;
      }>('repo.create', {
        title: 'Persian Workflow Demo',
        initial_payload: {
          user_problem_text: problemText,
          context: {},
          timeout_ms: 2000,
        },
      });

      if ('error' in createRes) {
        throw new Error(createRes.error.message);
      }

      const workflowRes = await rpc<WorkflowResult>('workflow.run', {
        repo_id: createRes.result.repo_id,
        workspace_id: createRes.result.head,
      });

      if ('error' in workflowRes) {
        throw new Error(workflowRes.error.message);
      }

      setResult(workflowRes.result);
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  const contradictionDetected = result?.explain_contradiction?.explanation;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex min-h-screen w-full max-w-5xl flex-col gap-6 px-4 py-8 sm:px-6 lg:px-8">
        <header className="space-y-3">
          <p className="text-sm font-medium text-slate-400">ProjectSolver AI</p>
          <h1 className="text-3xl font-extrabold tracking-tight sm:text-4xl">
            بررسی منطقی فارسی با Workflow خودکار
          </h1>
          <p className="max-w-3xl text-sm leading-7 text-slate-300 sm:text-base">
            متن فارسی را وارد کن، دکمه را بزن، و خروجی زنجیرهٔ
            formalize → check_consistency → explain_contradiction را ببین.
          </p>
        </header>

        <section className="grid gap-4 rounded-3xl border border-slate-800 bg-slate-900/70 p-4 shadow-2xl shadow-black/20 backdrop-blur sm:p-6">
          <label className="space-y-2">
            <span className="text-sm font-semibold text-slate-200">متن مسئله</span>
            <textarea
              value={problemText}
              onChange={(e) => setProblemText(e.target.value)}
              placeholder="مثلاً: پروژه الف نباید با پروژه ب همزمان باشد..."
              className="min-h-44 w-full rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-right leading-8 text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
            />
          </label>

          <div className="flex flex-wrap items-center gap-3">
            <button
              onClick={handleRun}
              disabled={loading}
              className="rounded-2xl bg-indigo-500 px-5 py-3 text-sm font-bold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loading ? 'در حال بررسی...' : 'بررسی منطقی'}
            </button>

            <span className="text-sm text-slate-400">یک کلیک برای اجرای کامل workflow</span>
          </div>
        </section>

        {rpcError ? (
          <section className="rounded-3xl border border-red-900/60 bg-red-950/70 p-5 text-red-100">
            <h2 className="mb-2 text-lg font-bold">خطا</h2>
            <p className="whitespace-pre-wrap leading-7">{rpcError}</p>
          </section>
        ) : null}

        {result ? (
          <section className="grid gap-4">
            <div className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
              <h2 className="mb-3 text-lg font-bold">وضعیت نهایی</h2>
              <p className="text-sm leading-7 text-slate-300">
                <span className="font-semibold text-slate-100">step:</span> {result.step}
              </p>
              <p className="text-sm leading-7 text-slate-300">
                <span className="font-semibold text-slate-100">status:</span> {result.status}
              </p>
              <p className="text-sm leading-7 text-slate-300">
                <span className="font-semibold text-slate-100">check:</span>{' '}
                {result.check_consistency?.sat?.status ?? 'unknown'}
              </p>
            </div>

            {contradictionDetected ? (
              <div className="rounded-3xl border border-rose-900/60 bg-rose-950/70 p-5">
                <h2 className="mb-3 text-lg font-bold text-rose-100">تناقض تشخیص داده شد</h2>
                {summary ? (
                  <p className="mb-4 whitespace-pre-wrap leading-8 text-rose-50">{summary}</p>
                ) : null}

                <div className="space-y-3">
                  {choices.map((choice) => (
                    <div
                      key={choice.choice_id}
                      className="rounded-2xl border border-rose-900/60 bg-rose-950/60 p-4"
                    >
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-rose-700 px-3 py-1 text-xs font-bold text-white">
                          {choice.choice_id}
                        </span>
                        <span className="text-xs text-rose-200">{choice.action}</span>
                        {choice.target_item_id ? (
                          <span className="text-xs text-rose-200">
                            target: {choice.target_item_id}
                          </span>
                        ) : null}
                      </div>
                      <p className="leading-7 text-rose-50">{choice.nl}</p>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="rounded-3xl border border-emerald-900/60 bg-emerald-950/70 p-5 text-emerald-50">
                <h2 className="mb-2 text-lg font-bold">نتیجه</h2>
                <p className="leading-7">
                  {result.check_consistency?.sat?.status === 'unsat'
                    ? 'تناقض تشخیص داده شد، اما توضیح ساختاریافته‌ای برنگشت.'
                    : 'Workflow با موفقیت اجرا شد و تناقضی گزارش نشد.'}
                </p>
              </div>
            )}

            <details className="rounded-3xl border border-slate-800 bg-slate-900/70 p-5">
              <summary className="cursor-pointer text-sm font-semibold text-slate-200">
                نمایش JSON خام
              </summary>
              <pre className="mt-4 overflow-auto rounded-2xl bg-slate-950/80 p-4 text-xs leading-6 text-slate-300">
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </section>
        ) : null}
      </div>
    </main>
  );
}
