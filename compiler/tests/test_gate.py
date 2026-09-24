import json,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from compiler.lib.runtime import RunError
from compiler.tests.run_tests import (REQUIRED_SUITES,all_passed,execute_suites,
                                     verify_source_unchanged,finish_report,command,check_unittest_result)


class GateTests(unittest.TestCase):
    def test_missing_suite_is_not_success(self):
        results=[dict(name=n,status='PASS') for n in REQUIRED_SUITES]
        self.assertTrue(all_passed(results))
        self.assertFalse(all_passed(results[:-1]));self.assertFalse(all_passed([]))
        self.assertFalse(all_passed(results+[results[0]]))
        for status in ('FAIL','SKIP','NOT_RUN'):
            results[0]['status']=status;self.assertFalse(all_passed(results))

    def test_failures_produce_nonzero_report(self):
        for failure in (subprocess.CompletedProcess(['fake'],2,'','failure'),
                        subprocess.TimeoutExpired(['fake'],1),FileNotFoundError('missing dependency')):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);records=[]
                def broken():
                    options={'side_effect':failure} if isinstance(failure,Exception) else {'return_value':failure}
                    with patch('compiler.tests.run_tests.subprocess.run',**options):
                        command(['fake'],root/'failure.log',records)
                    return {}
                actions={n:(lambda:{}) for n in REQUIRED_SUITES};actions['compiler']=broken
                report=dict(results=execute_suites(actions))
                self.assertNotEqual(finish_report(root,report),0)
                saved=json.loads((root/'report.json').read_text())
                self.assertTrue(next(x for x in saved['results'] if x['name']=='compiler')['reason'])

    def test_skip_empty_and_failure_unit_results_rejected(self):
        ok=dict(tests=3,failures=0,errors=0,skipped=0,expected_failures=0,unexpected_successes=0)
        check_unittest_result(ok)
        for key in ('failures','errors','skipped','expected_failures','unexpected_successes','tests'):
            bad=dict(ok);bad[key]=0 if key=='tests' else 1
            with self.assertRaises(RunError):check_unittest_result(bad)

    def test_source_changes_reject_release(self):
        verify_source_unchanged({'a':'123'},{'a':'123'})
        with self.assertRaises(RunError):verify_source_unchanged({'a':'123'},{'a':'456'})
        with self.assertRaises(RunError):verify_source_unchanged({'a':'123'},{'a':'123','b':'456'})
