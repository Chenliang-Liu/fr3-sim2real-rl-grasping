"""Hardware-free checks of the public runner's execution gates.

The actual PolicyCartesian class is extracted from its AST so these tests do not
import ROS, SAC, NumPy, or the bundled FR3Controller. Controller calls
are recorded. Immediate and deferred job stubs check both calls and scheduling.
These tests do not establish hardware safety or ROS/controller compatibility.
"""

import ast
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "fr3_sim2real" / "real_fr3_policy_cartesian.py"


class Vector(list):
    def __add__(self, other):
        return Vector(a + b for a, b in zip(self, other))

    def copy(self):
        return Vector(self)


class ImmediateThread:
    def __init__(self, target, args=(), daemon=False):
        self.target = target
        self.args = args

    def start(self):
        self.target(*self.args)


class DeferredThread(ImmediateThread):
    pending = []

    def start(self):
        self.pending.append(self)

    def run_pending(self):
        self.target(*self.args)


class Recorder:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append(("reset", (), {}))

    def move_to(self, *args, **kwargs):
        self.calls.append(("move_to", args, kwargs))

    def close_gripper(self, *args, **kwargs):
        self.calls.append(("close_gripper", args, kwargs))


def load_policy_class(thread_class=ImmediateThread):
    source = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
    class_node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCartesian")
    module = ast.Module(body=[class_node], type_ignores=[])
    fake_numpy = SimpleNamespace(
        asarray=lambda value, **kwargs: value,
        float32="float32",
        linalg=SimpleNamespace(norm=lambda values: sum(x * x for x in values) ** 0.5),
        array2string=lambda value, **kwargs: repr(value),
    )
    namespace = {
        "np": fake_numpy,
        "threading": SimpleNamespace(Thread=thread_class),
        "time": SimpleNamespace(time=lambda: 0.0, sleep=lambda seconds: None),
        "TransformException": RuntimeError,
        "quat_to_rpy": lambda *args: (0.0, 0.0, 0.0),
    }
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace["PolicyCartesian"]


PolicyCartesian = load_policy_class()


def make_runner(execute=False, enable_gripper=True, lift_after_grasp=True, reset=True, policy_class=PolicyCartesian):
    runner = policy_class.__new__(policy_class)
    runner.args = SimpleNamespace(
        execute=execute,
        enable_gripper=enable_gripper,
        lift_after_grasp=lift_after_grasp,
        reset=reset,
        model="test-model.zip",
        z_max=0.6,
        lift_height=0.075,
        grasp_width=0.0,
        gripper_speed=0.05,
        grasp_wait=0.0,
        close_action_threshold=-0.5,
        grasp_xy_threshold=0.035,
        grasp_height=0.085,
        print_hz=1.0,
    )
    logger = SimpleNamespace(warn=lambda *args, **kwargs: None, info=lambda *args, **kwargs: None, error=lambda *args, **kwargs: None)
    runner.node = SimpleNamespace(get_logger=lambda: logger)
    runner.controller = Recorder()
    runner.spin_thread = SimpleNamespace(start=lambda: None)
    runner.motion_lock = threading.Lock()
    runner.gripper_lock = threading.Lock()
    runner.grasping = False
    runner.grasped = False
    runner.lift_sent = False
    runner.last_print = 0.0
    runner.build_obs = lambda: ([], Vector([0.4, 0.0, 0.2]), Vector([0.4, 0.0, 0.15]), Vector([0.0, 0.0, -0.05]), [0.0, 0.0, 0.0, 1.0])
    runner.model = SimpleNamespace(predict=lambda *args, **kwargs: ([0.1, 0.0, -0.1, -0.9], None))
    runner.action_to_delta = lambda *args: Vector([0.001, 0.0, -0.001])
    return runner


class ExecutionGateTests(unittest.TestCase):
    def test_dry_run_with_all_motion_flags_never_calls_controller(self):
        runner = make_runner()
        runner.start()
        runner.step()
        runner.close_gripper_sequence()
        runner.send_motion(Vector([0.4, 0.0, 0.2]), [0.0, 0.0, 0.0])
        runner.send_lift(Vector([0.4, 0.0, 0.2]), [0.0, 0.0, 0.0])
        self.assertEqual(runner.controller.calls, [])
        self.assertFalse(runner.grasped)
        self.assertFalse(runner.lift_sent)

    def test_dry_run_does_not_schedule_lift_even_if_grasp_flag_is_set(self):
        runner = make_runner()
        runner.grasped = True
        scheduled = []
        runner.send_lift = lambda *args: scheduled.append(args)
        runner.step()
        self.assertEqual(scheduled, [])
        self.assertEqual(runner.controller.calls, [])

    def test_dry_run_does_not_schedule_any_motion_worker(self):
        policy_class = load_policy_class(thread_class=DeferredThread)
        for grasped in (False, True):
            with self.subTest(grasped=grasped):
                DeferredThread.pending = []
                runner = make_runner(policy_class=policy_class)
                runner.grasped = grasped
                runner.step()
                self.assertEqual(DeferredThread.pending, [])
                self.assertEqual(runner.controller.calls, [])

    def test_queued_workers_recheck_execution_at_entry(self):
        policy_class = load_policy_class(thread_class=DeferredThread)
        for phase in ("approach", "close", "lift"):
            with self.subTest(phase=phase):
                DeferredThread.pending = []
                runner = make_runner(execute=True, enable_gripper=phase == "close", reset=False, policy_class=policy_class)
                runner.grasped = phase == "lift"
                runner.step()
                self.assertTrue(DeferredThread.pending)
                runner.args.execute = False
                for job in DeferredThread.pending:
                    job.run_pending()
                self.assertEqual(runner.controller.calls, [])
                self.assertFalse(runner.lift_sent)
                self.assertEqual(runner.grasped, phase == "lift")

    def test_execute_preserves_reset_and_approach_motion(self):
        runner = make_runner(execute=True, enable_gripper=False)
        runner.start()
        runner.step()
        self.assertEqual([call[0] for call in runner.controller.calls], ["reset", "move_to"])
        self.assertTrue(runner.controller.calls[1][2]["execute"])
        self.assertFalse(runner.grasped)

    def test_execute_preserves_gripper_then_fixed_lift(self):
        runner = make_runner(execute=True, reset=False)
        runner.step()
        self.assertEqual([call[0] for call in runner.controller.calls], ["close_gripper", "move_to"])
        self.assertTrue(runner.grasped)
        self.assertTrue(runner.lift_sent)
        self.assertAlmostEqual(runner.controller.calls[1][1][2], 0.275)
        self.assertTrue(runner.controller.calls[1][2]["execute"])

    def test_gripper_flag_is_required_even_with_execution_enabled(self):
        runner = make_runner(execute=True, enable_gripper=False)
        runner.close_gripper_sequence()
        self.assertEqual(runner.controller.calls, [])
        self.assertFalse(runner.grasped)


if __name__ == "__main__":
    unittest.main()
