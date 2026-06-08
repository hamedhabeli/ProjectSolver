use std::{
    sync::{
        mpsc::{self, Receiver},
        Mutex,
    },
    time::Duration,
};

use tauri::{AppHandle, Manager, State};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

struct BridgeState {
    child: Mutex<CommandChild>,
    stdout_rx: Mutex<Receiver<String>>,
}

fn spawn_core_sidecar(app: &AppHandle) -> BridgeState {
    let command = app
        .shell()
        .sidecar("core_py")
        .expect("failed to resolve sidecar 'core_py'");

    let (mut rx, child) = command.spawn().expect("failed to spawn sidecar 'core_py'");

    let (tx, stdout_rx) = mpsc::channel::<String>();

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).to_string();
                    let _ = tx.send(line);
                }
                CommandEvent::Stderr(bytes) => {
                    eprintln!("[core_py stderr] {}", String::from_utf8_lossy(&bytes));
                }
                _ => {}
            }
        }
    });

    BridgeState {
        child: Mutex::new(child),
        stdout_rx: Mutex::new(stdout_rx),
    }
}

#[tauri::command]
fn rpc_call(state: State<'_, BridgeState>, request_json: String) -> Result<String, String> {
    let line = if request_json.ends_with('\n') {
        request_json
    } else {
        format!("{request_json}\n")
    };

    {
        let mut child = state
            .child
            .lock()
            .map_err(|e| format!("sidecar lock error: {e}"))?;
        child
            .write(line.as_bytes())
            .map_err(|e| format!("failed to write to sidecar stdin: {e}"))?;
    }

    let rx = state
        .stdout_rx
        .lock()
        .map_err(|e| format!("stdout lock error: {e}"))?;

    let response = rx
        .recv_timeout(Duration::from_secs(30))
        .map_err(|e| format!("timed out waiting for sidecar response: {e}"))?;

    Ok(response)
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            let bridge = spawn_core_sidecar(&app.handle());
            app.manage(bridge);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![rpc_call])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
