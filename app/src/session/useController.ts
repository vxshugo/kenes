import { useEffect, useSyncExternalStore } from "react";
import { SessionController, type ControllerState } from "./controller";

export const controller = new SessionController();

export function useController(): ControllerState {
  useEffect(() => {
    void controller.init();
  }, []);
  return useSyncExternalStore(controller.subscribe, controller.getState, controller.getState);
}
