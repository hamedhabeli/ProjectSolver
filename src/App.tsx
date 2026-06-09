import { useEffect, useMemo, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';

type RpcError = { code: number; message: string; data?: unknown };
type RpcSuccess<T> = { jsonrpc: '2.0'; id: string; result: T };
type RpcFailure = { jsonrpc: '2.0'; id: string; error: RpcError };
type RpcEnvelope<T> = RpcSuccess<T> | RpcFailure;

type GeminiModelInfo = {
  name: string;
  display_name: string;
  description?: string;
  supported_generation_methods?: string[];
};

type LlmConfig = {
  provider: 'gemini';
  api_key: string;
  model: string;
  temperature: number;
  top_p: number;
  timeout_s?: number;
  max_retries?: number;
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
      syntax: { valid: boolean; errors: Array<{ code: string; message: string; detail?: string }> };
    }>;
  };
  check_consistency?: {
    step: string;
    sat: { status: 'sat' | 'unsat' | 'unknown'; reason_unknown?: string | null };
    unsat_core_item_ids?: string[];
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

function unwrap<T>(envelope: RpcEnvelope<T>): T {
  if ('error' in envelope) {
    throw new Error(envelope.error.message);
  }
  return envelope.result;
}

async function rpc<T>(method: string, params: Record<string, unknown> = {}): Promise<RpcEnvelope<T>> {
  const request = {
    jsonrpc: '2.0',
    id: newId(),
    method,
    params,
  };
  const responseJson = await invoke<string>('rpc_call', {
    requestJson: JSON.stringify(request),
  });
  return JSON.parse(responseJson) as RpcEnvelope<T>;
}

const DEFAULT_PROBLEM = `پروژه الف نباید با پروژه ب همزمان فعال باشد.
همچنین پروژه الف باید همزمان فعال باشد.`;

export default function App() {
  const [problemText, setProblemText] = useState(DEFAULT_PROBLEM);

  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('');
  const [temperature, setTemperature] = useState('0.2');
  const [topP, setTopP] = useState('0.95');
  const [models, setModels] = useState<GeminiModelInfo[]>([]);

  const [loadingSettings, setLoadingSettings] = useState(true);
  const [loadingModels, setLoadingModels] = useState(false);
  const [savingSettings, setSavingSettings] = useState(false);
  const [testingConnection, setTestingConnection] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);

  const [rpcError, setRpcError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [result, setResult] = useState<WorkflowResult | null>(null);

  const summary = useMemo(() => {
    if (!result) return null;
    return result.explain_contradiction?.explanation?.nl_summary ?? null;
  }, [result]);

  const choices = useMemo(() => {
    if (!result) return [];
    return result.explain_contradiction?.explanation?.choices ?? [];
  }, [result]);

  const parsedTemperature = Number.parseFloat(temperature);
  const parsedTopP = Number.parseFloat(topP);
  const canRun =
    apiKey.trim().length > 0 &&
    model.trim().length > 0 &&
    Number.isFinite(parsedTemperature) &&
    Number.isFinite(parsedTopP) &&
    !loadingSettings &&
    !loadingModels &&
    !savingSettings &&
    !testingConnection &&
    !loadingRun;

  useEffect(() => {
    void loadSavedSettings();
  }, []);

  async function loadSavedSettings() {
    setLoadingSettings(true);
    setRpcError(null);
    try {
      const response = await rpc<LlmConfig>('llm.get_config');
      const config = unwrap(response);
      setApiKey(config.api_key ?? '');
      setModel(config.model ?? '');
      setTemperature(String(config.temperature ?? 0.2));
      setTopP(String(config.top_p ?? 0.95));

      if (config.api_key) {
        await loadModels(config.api_key, config.model);
      }
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingSettings(false);
    }
  }

  async function loadModels(apiKeyOverride?: string, preferredModel?: string) {
    const key = (apiKeyOverride ?? apiKey).trim();
    if (!key) {
      setModels([]);
      setStatusMessage('برای دریافت مدل‌ها، Gemini API Key را وارد کن.');
      return false;
    }

    setLoadingModels(true);
    setRpcError(null);
    setStatusMessage(null);
    try {
      const response = await rpc<{ models: GeminiModelInfo[] }>('llm.gemini.list_models', {
        api_key: key,
        timeout_s: 30,
      });
      const payload = unwrap(response);
      const nextModels = payload.models ?? [];
      setModels(nextModels);

      const preferred =
        preferredModel && nextModels.some((m) => m.name === preferredModel)
          ? preferredModel
          : nextModels[0]?.name ?? '';

      if (preferred) {
        setModel(preferred);
      } else if (model && !nextModels.some((m) => m.name === model)) {
        setModel('');
      }

      setStatusMessage(`تعداد ${nextModels.length} مدل Gemini بارگذاری شد.`);
      return true;
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setLoadingModels(false);
    }
  }

  async function saveSettings() {
    const apiKeyTrimmed = apiKey.trim();
    const modelTrimmed = model.trim();

    if (!apiKeyTrimmed) {
      setRpcError('Gemini API Key را وارد کن.');
      return false;
    }
    if (!modelTrimmed) {
      setRpcError('یک مدل انتخاب کن.');
      return false;
    }
    if (!Number.isFinite(parsedTemperature)) {
      setRpcError('Temperature نامعتبر است.');
      return false;
    }
    if (!Number.isFinite(parsedTopP)) {
      setRpcError('Top-p نامعتبر است.');
      return false;
    }

    setSavingSettings(true);
    setRpcError(null);
    setStatusMessage(null);
    try {
      const response = await rpc<LlmConfig>('llm.set_config', {
        config: {
          provider: 'gemini',
          api_key: apiKeyTrimmed,
          model: modelTrimmed,
          temperature: parsedTemperature,
          top_p: parsedTopP,
          timeout_s: 30,
          max_retries: 2,
        },
      });
      const saved = unwrap(response);
      setApiKey(saved.api_key ?? apiKeyTrimmed);
      setModel(saved.model ?? modelTrimmed);
      setTemperature(String(saved.temperature ?? parsedTemperature));
      setTopP(String(saved.top_p ?? parsedTopP));
      setStatusMessage('تنظیمات Gemini ذخیره شد.');
      return true;
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
      return false;
    } finally {
      setSavingSettings(false);
    }
  }

  async function testConnection() {
    const apiKeyTrimmed = apiKey.trim();
    if (!apiKeyTrimmed) {
      setRpcError('ابتدا Gemini API Key را وارد کن.');
      return;
    }

    setTestingConnection(true);
    setRpcError(null);
    setStatusMessage(null);
    try {
      const response = await rpc<{ ok: boolean; models_count: number }>('llm.gemini.test_key', {
        api_key: apiKeyTrimmed,
        timeout_s: 30,
      });
      const payload = unwrap(response);
      setStatusMessage(`اتصال برقرار شد. ${payload.models_count} مدل قابل استفاده پیدا شد.`);
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
    } finally {
      setTestingConnection(false);
    }
  }

  async function handleRun() {
    setLoadingRun(true);
    setRpcError(null);
    setResult(null);
    setStatusMessage(null);

    try {
      const saved = await saveSettings();
      if (!saved) return;

      const createRes = await rpc<{ repo_id: string; head: string }>('repo.create', {
        title: 'Persian Workflow Demo',
        initial_payload: {
          user_problem_text: problemText,
          context: {},
          timeout_ms: 2000,
        },
      });
      const created = unwrap(createRes);

      const workflowRes = await rpc<WorkflowResult>('workflow.run', {
        repo_id: created.repo_id,
        workspace_id: created.head,
      });
      const workflow = unwrap(workflowRes);
      setResult(workflow);
    } catch (err) {
      setRpcError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingRun(false);
    }
  }

  const selectedModelInfo = useMemo(
    () => models.find((m) => m.name === model) ?? null,
    [models, model],
  );

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 p-6">
        <header className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/40">
          <h1 className="text-3xl font-black tracking-tight text-white">ProjectSolver AI</h1>
          <p className="mt-2 max-w-3xl leading-8 text-slate-300">
            تنظیم Gemini، انتخاب مدل، تنظیم temperature و top-p، و اجرای workflow منطقی از همین صفحه.
          </p>
        </header>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/40">
          <div className="mb-5 flex items-center justify-between gap-3">
            <div>
              <h2 className="text-xl font-bold text-white">تنظیمات Gemini</h2>
              <p className="mt-1 text-sm text-slate-400">
                API Key بین اجراها روی سرور ذخیره می‌شود.
              </p>
            </div>
            {loadingSettings ? <span className="text-sm text-slate-400">در حال بارگذاری...</span> : null}
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            <label className="grid gap-2">
              <span className="text-sm font-semibold text-slate-200">Gemini API Key</span>
              <input
                id="apiKey"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="AIza..."
                className="rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-left text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-semibold text-slate-200">Model</span>
              <select
                id="model"
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
              >
                <option value="">ابتدا Load models را بزن</option>
                {models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.display_name || m.name}
                  </option>
                ))}
              </select>
              {selectedModelInfo ? (
                <p className="text-xs leading-6 text-slate-400">
                  {selectedModelInfo.description || 'بدون توضیح'}
                </p>
              ) : null}
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-semibold text-slate-200">Temperature</span>
              <input
                id="temperature"
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
              />
            </label>

            <label className="grid gap-2">
              <span className="text-sm font-semibold text-slate-200">Top-p</span>
              <input
                id="topP"
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={topP}
                onChange={(e) => setTopP(e.target.value)}
                className="rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
              />
            </label>
          </div>

          <div className="mt-5 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => void loadModels()}
              disabled={loadingModels || loadingSettings}
              className="rounded-2xl bg-slate-700 px-5 py-3 text-sm font-bold text-white transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingModels ? 'در حال دریافت مدل‌ها...' : 'Load models'}
            </button>
            <button
              type="button"
              onClick={() => void testConnection()}
              disabled={testingConnection || loadingSettings}
              className="rounded-2xl bg-cyan-600 px-5 py-3 text-sm font-bold text-white transition hover:bg-cyan-500 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {testingConnection ? 'در حال تست...' : 'Test connection'}
            </button>
            <button
              type="button"
              onClick={() => void saveSettings()}
              disabled={savingSettings || loadingSettings}
              className="rounded-2xl bg-indigo-500 px-5 py-3 text-sm font-bold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {savingSettings ? 'در حال ذخیره...' : 'Save settings'}
            </button>
            {statusMessage ? <span className="text-sm text-slate-400">{statusMessage}</span> : null}
          </div>
        </section>

        <section className="rounded-3xl border border-slate-800 bg-slate-900/70 p-6 shadow-xl shadow-slate-950/40">
          <h2 className="mb-4 text-xl font-bold text-white">مسئله فارسی</h2>
          <label className="grid gap-2">
            <span className="text-sm font-semibold text-slate-200">متن مسئله</span>
            <textarea
              value={problemText}
              onChange={(e) => setProblemText(e.target.value)}
              placeholder="مثلاً: پروژه الف نباید با پروژه ب همزمان باشد..."
              className="min-h-44 w-full rounded-2xl border border-slate-700 bg-slate-950/80 p-4 text-right leading-8 text-slate-100 outline-none transition focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/40"
            />
          </label>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button
              onClick={() => void handleRun()}
              disabled={!canRun}
              className="rounded-2xl bg-emerald-500 px-5 py-3 text-sm font-bold text-white transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {loadingRun ? 'در حال بررسی...' : 'بررسی منطقی'}
            </button>
            <span className="text-sm text-slate-400">workflow کامل: formalize → check_consistency → explain_contradiction</span>
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

            {result.explain_contradiction?.explanation ? (
              <div className="rounded-3xl border border-rose-900/60 bg-rose-950/70 p-5">
                <h2 className="mb-3 text-lg font-bold text-rose-100">تناقض تشخیص داده شد</h2>
                {summary ? (
                  <p className="mb-4 whitespace-pre-wrap leading-8 text-rose-50">{summary}</p>
                ) : null}
                <div className="space-y-3">
                  {choices.map((choice) => (
                    <div key={choice.choice_id} className="rounded-2xl border border-rose-900/60 bg-rose-950/60 p-4">
                      <div className="mb-2 flex flex-wrap items-center gap-2">
                        <span className="rounded-full bg-rose-700 px-3 py-1 text-xs font-bold text-white">
                          {choice.choice_id}
                        </span>
                        <span className="text-xs text-rose-200">{choice.action}</span>
                        {choice.target_item_id ? (
                          <span className="text-xs text-rose-200">target: {choice.target_item_id}</span>
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
              <summary className="cursor-pointer text-sm font-semibold text-slate-200">نمایش JSON خام</summary>
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