import plistlib
import subprocess
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class LaunchAgentTests(unittest.TestCase):
    def test_install_script_renders_launchagent_for_current_checkout(self):
        installer = PROJECT_ROOT / "scripts" / "install-launchagent.sh"

        result = subprocess.run(
            [str(installer), "--print-plist"],
            check=True,
            capture_output=True,
        )
        plist = plistlib.loads(result.stdout)

        self.assertEqual(plist["Label"], "com.kkonstant.ytranslate")
        self.assertEqual(
            plist["ProgramArguments"],
            [str(PROJECT_ROOT / "scripts" / "run-ytranslate-server.sh")],
        )
        self.assertEqual(plist["WorkingDirectory"], str(PROJECT_ROOT))
        self.assertTrue(plist["RunAtLoad"])
        self.assertTrue(plist["KeepAlive"])
        self.assertEqual(plist["EnvironmentVariables"]["PYTHONUNBUFFERED"], "1")
        self.assertIn("/Users/kkonstant/.local/bin", plist["EnvironmentVariables"]["PATH"])
        self.assertEqual(
            plist["StandardOutPath"],
            "/Users/kkonstant/Library/Logs/ytranslate/server.log",
        )
        self.assertEqual(
            plist["StandardErrorPath"],
            "/Users/kkonstant/Library/Logs/ytranslate/server.err.log",
        )
