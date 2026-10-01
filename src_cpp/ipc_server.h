#pragma once

// Unix-domain-socket bridge to env_python/ipc_client.py, native (macOS/Linux) build only -
// the Wasm build uses state_export_wasm.cpp instead, which has no sockets.
namespace IpcServer
{
	// Reads the PINBALL_IPC_SOCK environment variable. If unset, IPC stays disabled and
	// SyncTick() is a no-op, so the binary runs as a normal, unmanaged build for manual
	// play/debugging. If set, binds that path and blocks until one RL client connects.
	void Init();

	void Shutdown();

	// Call once per tick from pb::timed_frame(), after that tick's physics has been applied.
	// No-op when IPC is disabled. Otherwise: sends the tick's StateFrame, blocks for the
	// client's ActionFrame, and applies it (flipper edges, launch/reset pulses) before returning -
	// this is what makes the ACTION EXCHANGE lockstep rather than wall-clock paced. Physics dt
	// itself is a separate concern - see IsEnabled()'s use in winmain.cpp.
	void SyncTick(float timeDeltaSec);

	// True once Init() has bound a client. winmain.cpp uses this to decide whether the physics
	// timestep should come from a fixed nominal value instead of measured wall-clock frame time -
	// see the dt override at its MainLoop() call site.
	bool IsEnabled();
}
