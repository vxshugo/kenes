//! Stub backend for platforms other than Linux and macOS.

use anyhow::{bail, Result};
use kenes_types::DeviceInfo;

use crate::{DeviceSel, StreamCtx, StreamInfo};

const UNSUPPORTED: &str = "audio capture is only implemented for Linux and macOS";

pub(crate) struct Worker;

impl Worker {
    pub(crate) fn signal_stop(&mut self) {}
    pub(crate) fn join(&mut self) {}
}

pub(crate) fn list_devices() -> Result<Vec<DeviceInfo>> {
    bail!(UNSUPPORTED)
}

pub(crate) fn open_stream(_sel: &DeviceSel, _ctx: StreamCtx) -> Result<(Worker, StreamInfo)> {
    bail!(UNSUPPORTED)
}
