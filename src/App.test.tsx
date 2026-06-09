import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import App from './App';

const invokeMock = vi.fn();

vi.mock('@tauri-apps/api/core', () => ({
  invoke: (...args: unknown[]) => invokeMock(...args),
}));

function ok<T>(id: string, result: T) {
  return JSON.stringify({ jsonrpc: '2.0', id, result });
}

function rpcFromArgs(args: any) {
  return JSON.parse(args.requestJson) as { id: string; method: string; params: Record<string, unknown> };
}

beforeEach(() => {
  invokeMock.mockReset();
  invokeMock.mockImplementation(async (_cmd: string, args: { requestJson: string }) => {
    const req = rpcFromArgs(args);

    switch (req.method) {
      case 'llm.get_config':
        return ok(req.id, {
          provider: 'gemini',
          api_key: 'test-key',
          model: 'models/gemini-2.0-flash',
          temperature: 0.2,
          top_p: 0.95,
          timeout_s: 30,
          max_retries: 2,
        });

      case 'llm.gemini.list_models':
        return ok(req.id, {
          models: [
            {
              name: 'models/gemini-2.0-flash',
              display_name: 'Gemini Flash',
              description: 'fast',
              supported_generation_methods: ['generateContent'],
            },
          ],
        });

      case 'llm.set_config':
        return ok(req.id, req.params.config);

      case 'llm.gemini.test_key':
        return ok(req.id, { ok: true, models_count: 1 });

      case 'repo.create':
        return ok(req.id, { repo_id: 'R_1', head: 'N_1' });

      case 'workflow.run':
        return ok(req.id, {
          step: 'workflow.run',
          status: 'ok',
          check_consistency: { step: 'check_consistency', sat: { status: 'sat' } },
          explain_contradiction: null,
        });

      default:
        throw new Error(`unexpected method: ${req.method}`);
    }
  });
});

describe('App', () => {
  it('loads persisted Gemini settings and models', async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByLabelText(/Gemini API Key/i)).toHaveValue('test-key');
    });

    expect(screen.getByLabelText(/Model/i)).toHaveValue('models/gemini-2.0-flash');
    expect(screen.getByRole('option', { name: /Gemini Flash/i })).toBeInTheDocument();
  });

  it('sends requestJson to rpc_call and runs workflow', async () => {
    const user = userEvent.setup();
    render(<App />);

    await waitFor(() => {
      expect(screen.getByLabelText(/Gemini API Key/i)).toHaveValue('test-key');
    });

    await user.click(screen.getByRole('button', { name: /بررسی منطقی/i }));

    await waitFor(() => {
      expect(invokeMock).toHaveBeenCalled();
    });

    for (const call of invokeMock.mock.calls) {
      expect(call[1]).toHaveProperty('requestJson');
    }

    expect(screen.getByText(/Workflow with success|Workflow با موفقیت|Workflow با موفقیت اجرا شد/i)).toBeInTheDocument();
  });
});