"""Exercise process guards using only subprocesses created by these tests."""

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest


spec = importlib.util.spec_from_file_location(
    "cache_switch", Path(__file__).resolve().parents[1] / "scripts/switch_cached_training.py")
switch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(switch)


@unittest.skipUnless(sys.platform == "linux", "Linux /proc process controls")
class SwitchGuardTests(unittest.TestCase):
    def test_identity_guard_and_pause_resume(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            started = switch.identity(process.pid)
            self.assertIsNotNone(started)
            switch.send_verified({process.pid: "incorrect-start-time"}, signal.SIGKILL)
            self.assertIsNone(process.poll())
            switch.send_verified({process.pid: started}, signal.SIGSTOP)
            for _ in range(100):
                state = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "T":
                    break
                time.sleep(0.01)
            self.assertEqual(state, "T")
            switch.send_verified({process.pid: started}, signal.SIGCONT)
            self.assertIsNone(process.poll())
        finally:
            process.kill()
            process.wait()

    def test_descendants_does_not_include_unrelated_process(self):
        parent = subprocess.Popen([sys.executable, "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); time.sleep(30)"])
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        children = {}
        try:
            for _ in range(100):
                children = switch.descendants(parent.pid)
                if children:
                    break
                time.sleep(0.01)
            self.assertTrue(children)
            self.assertNotIn(unrelated.pid, children)
            switch.terminate({parent.pid: switch.identity(parent.pid), **children})
            parent.wait(timeout=5)
            self.assertIsNone(unrelated.poll())
        finally:
            switch.send_verified(children, signal.SIGKILL)
            for process in (parent, unrelated):
                if process.poll() is None:
                    process.kill()
                process.wait()


if __name__ == "__main__":
    unittest.main()
