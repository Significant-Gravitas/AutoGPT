import gc
import unittest

from startup_gc_control import observe_startup


class StartupGcControlTest(unittest.TestCase):
    def test_restores_original_state_and_callbacks_even_on_import_error(self):
        original = gc.isenabled()
        callbacks = list(gc.callbacks)
        try:
            for initial in (True, False):
                for disabled in (True, False):
                    for raises in (True, False):
                        with self.subTest(
                            initial=initial, disabled=disabled, raises=raises
                        ):
                            gc.enable() if initial else gc.disable()
                            try:
                                with observe_startup(disabled) as result:
                                    self.assertEqual(
                                        gc.isenabled(), initial and not disabled
                                    )
                                    gc.collect(0)
                                    if raises:
                                        raise ImportError("failed startup")
                            except ImportError as error:
                                self.assertEqual(str(error), "failed startup")
                                self.assertTrue(raises)
                            self.assertEqual(gc.isenabled(), initial)
                            self.assertEqual(gc.callbacks, callbacks)
                            self.assertEqual(result["finally_enabled"], initial)
                            self.assertGreaterEqual(len(result["collections"]), 1)
                            self.assertGreaterEqual(result["gc_seconds"], 0)
                            self.assertGreater(result["seconds"], 0)
        finally:
            gc.enable() if original else gc.disable()


if __name__ == "__main__":
    unittest.main()
